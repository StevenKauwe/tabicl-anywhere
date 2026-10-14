//! Reading a CSV table into numbers the way the reference treats a data frame: numeric columns stay numbers
//! (empty, `NA`, `NaN`, `?` mean missing), any other column is a category and becomes the rank of its value among
//! the training values in sorted order (missing or never-seen values become -1). Categorical columns are placed
//! before numeric ones, as the reference's column transformer does.

use std::io;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Split CSV text into rows of fields. Handles quoted fields with doubled quotes; a field may contain newlines.
pub fn parse(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let (mut row, mut field, mut quoted) = (Vec::new(), String::new(), false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            ('"', _) => quoted = !quoted,
            (',', false) => row.push(std::mem::take(&mut field)),
            ('\n', false) => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            ('\r', false) => {}
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows.retain(|r| !(r.len() == 1 && r[0].is_empty()));
    rows
}

fn missing(s: &str) -> bool {
    let s = s.trim();
    s.is_empty() || s == "?" || s.eq_ignore_ascii_case("na") || s.eq_ignore_ascii_case("nan") || s.eq_ignore_ascii_case("null")
}

/// How a table's columns turn into numbers, learned from the training file and stored with the fitted model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Schema {
    pub target: String,
    /// The target's distinct values in class-index order.
    pub labels: Vec<String>,
    /// Feature columns in model order: name, and for a categorical column its sorted categories.
    pub columns: Vec<(String, Option<Vec<String>>)>,
}

impl Schema {
    pub fn to_text(&self) -> String {
        let mut s = format!("target\t{}\n", self.target);
        for l in &self.labels {
            s += &format!("label\t{l}\n");
        }
        for (name, cats) in &self.columns {
            match cats {
                None => s += &format!("num\t{name}\n"),
                Some(c) => s += &format!("cat\t{name}\t{}\n", c.join("\t")),
            }
        }
        s
    }

    pub fn from_text(text: &str) -> Schema {
        let mut s = Schema::default();
        for line in text.lines() {
            let mut p = line.split('\t');
            match (p.next(), p.next()) {
                (Some("target"), Some(t)) => s.target = t.to_string(),
                (Some("label"), Some(l)) => s.labels.push(l.to_string()),
                (Some("num"), Some(n)) => s.columns.push((n.to_string(), None)),
                (Some("cat"), Some(n)) => s.columns.push((n.to_string(), Some(p.map(|c| c.to_string()).collect()))),
                _ => {}
            }
        }
        s
    }

    /// Learn the schema from training rows (first row is the header) and return it with the features
    /// (`[rows][columns]`) and class indices.
    pub fn learn(rows: &[Vec<String>], target: &str) -> io::Result<(Schema, Vec<f64>, Vec<usize>)> {
        let head = rows.first().ok_or_else(|| bad("empty table".into()))?;
        let ti = head.iter().position(|h| h.trim() == target).ok_or_else(|| bad(format!("no column named {target}; columns are {head:?}")))?;
        let body = &rows[1..];
        if let Some((i, r)) = body.iter().enumerate().find(|(_, r)| r.len() != head.len()) {
            return Err(bad(format!("row {} has {} fields, the header has {}", i + 2, r.len(), head.len())));
        }
        // labels: sorted as numbers if they all are numbers, as text otherwise
        let mut labels: Vec<String> = body.iter().map(|r| r[ti].trim().to_string()).collect();
        if labels.iter().any(|l| missing(l)) {
            return Err(bad("the target column has missing values".into()));
        }
        let numeric = labels.iter().all(|l| l.parse::<f64>().is_ok());
        labels.sort_by(|a, b| if numeric { a.parse::<f64>().unwrap().total_cmp(&b.parse::<f64>().unwrap()) } else { a.cmp(b) });
        labels.dedup();
        let (mut cats, mut nums) = (Vec::new(), Vec::new());
        for (j, name) in head.iter().enumerate().filter(|(j, _)| *j != ti) {
            let is_num = body.iter().all(|r| missing(&r[j]) || r[j].trim().parse::<f64>().is_ok());
            if is_num {
                nums.push((j, name.trim().to_string(), None));
            } else {
                let mut c: Vec<String> = body.iter().filter(|r| !missing(&r[j])).map(|r| r[j].trim().to_string()).collect();
                c.sort();
                c.dedup();
                cats.push((j, name.trim().to_string(), Some(c)));
            }
        }
        let schema = Schema { target: target.to_string(), labels, columns: cats.into_iter().chain(nums).map(|(_, n, c)| (n, c)).collect() };
        let x = schema.features(rows)?;
        let y = body.iter().map(|r| schema.labels.iter().position(|l| l == r[ti].trim()).unwrap()).collect();
        Ok((schema, x, y))
    }

    /// For a numeric target (regression): the schema (no labels), the features and the target values.
    pub fn learn_numeric(rows: &[Vec<String>], target: &str) -> io::Result<(Schema, Vec<f64>, Vec<f64>)> {
        let head = rows.first().ok_or_else(|| bad("empty table".into()))?;
        let ti = head.iter().position(|h| h.trim() == target).ok_or_else(|| bad(format!("no column named {target}; columns are {head:?}")))?;
        let y = rows[1..]
            .iter()
            .enumerate()
            .map(|(i, r)| r.get(ti).and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite()).ok_or_else(|| bad(format!("row {}: the target is not a number", i + 2))))
            .collect::<io::Result<Vec<f64>>>()?;
        // learn the columns as usual, with every row given the same stand-in label
        let mut relabelled = rows.to_vec();
        relabelled[1..].iter_mut().for_each(|r| r[ti] = "0".into());
        let (mut schema, x, _) = Schema::learn(&relabelled, target)?;
        schema.labels.clear();
        Ok((schema, x, y))
    }

    /// Features (`[rows][columns]`) of a table with a header, by column name; the target column need not be there.
    pub fn features(&self, rows: &[Vec<String>]) -> io::Result<Vec<f64>> {
        let head = rows.first().ok_or_else(|| bad("empty table".into()))?;
        let at: Vec<usize> = self.columns.iter().map(|(n, _)| head.iter().position(|h| h.trim() == n).ok_or_else(|| bad(format!("the table has no column named {n}")))).collect::<io::Result<_>>()?;
        let mut x = Vec::with_capacity((rows.len() - 1) * at.len());
        for (i, r) in rows[1..].iter().enumerate() {
            for (&j, (name, cats)) in at.iter().zip(&self.columns) {
                let cell = r.get(j).map(|s| s.trim()).unwrap_or("");
                x.push(match cats {
                    Some(c) => c.binary_search_by(|v| v.as_str().cmp(cell)).map(|k| k as f64).unwrap_or(-1.0),
                    None if missing(cell) => f64::NAN,
                    None => cell.parse::<f64>().map_err(|_| bad(format!("row {}, column {name}: '{cell}' is not a number (the column was numeric when fitting)", i + 2)))?,
                });
            }
        }
        Ok(x)
    }
}

/// What a column of a table looks like, for showing to a person before they choose a target.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnInfo {
    pub name: String,
    /// Every non-missing cell reads as a number.
    pub numeric: bool,
    /// Distinct non-missing values.
    pub distinct: usize,
    pub missing: usize,
    /// The first few distinct values, in order of appearance.
    pub examples: Vec<String>,
}

/// Describe each column of parsed rows (first row is the header), by the same rules `Schema::learn` applies.
pub fn describe(rows: &[Vec<String>]) -> io::Result<Vec<ColumnInfo>> {
    let head = rows.first().ok_or_else(|| bad("the table is empty".into()))?;
    let body = &rows[1..];
    if let Some((i, r)) = body.iter().enumerate().find(|(_, r)| r.len() != head.len()) {
        return Err(bad(format!("row {} has {} fields, the header has {}", i + 2, r.len(), head.len())));
    }
    for (j, name) in head.iter().enumerate() {
        if head[..j].iter().any(|h| h.trim() == name.trim()) {
            return Err(bad(format!("two columns are both named '{}'; column names must differ", name.trim())));
        }
    }
    Ok(head
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let cells: Vec<&str> = body.iter().map(|r| r[j].trim()).filter(|c| !missing(c)).collect();
            let mut seen = std::collections::HashSet::new();
            let mut examples = Vec::new();
            for c in &cells {
                if seen.insert(*c) && examples.len() < 4 {
                    examples.push(c.to_string());
                }
            }
            ColumnInfo { name: name.trim().to_string(), numeric: cells.iter().all(|c| c.parse::<f64>().is_ok()), distinct: seen.len(), missing: body.len() - cells.len(), examples }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_and_reuse() {
        let rows = parse("a,kind,y\n1.5,red,yes\n,\"blue, dark\",no\n3,red,yes\n");
        let (s, x, y) = Schema::learn(&rows, "y").unwrap();
        assert_eq!(s.labels, ["no", "yes"]);
        assert_eq!(y, [1, 0, 1]);
        // categorical column first; "blue, dark" < "red"
        assert_eq!(x[0], 1.0);
        assert_eq!(x[2], 0.0);
        assert!(x[3].is_nan());
        assert_eq!(Schema::from_text(&s.to_text()), s);
        let t = s.features(&parse("kind,a\ngreen,2\n")).unwrap();
        assert_eq!(t, [-1.0, 2.0]);
        let d = describe(&rows).unwrap();
        assert_eq!((d[0].numeric, d[0].distinct, d[0].missing), (true, 2, 1));
        assert_eq!((d[1].numeric, d[1].distinct, d[1].examples.len()), (false, 2, 2));
        assert!(describe(&parse("a,a\n1,2\n")).is_err());
    }
}
