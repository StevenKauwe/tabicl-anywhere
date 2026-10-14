/* @ts-self-types="./tabicl_anywhere_wasm.d.ts" */

/**
 * A graphics card, through the browser's WebGPU: the device, its tier for fitted models (one tenant: this
 * page) and its working memory.
 */
export class Card {
    static __wrap(ptr) {
        const obj = Object.create(Card.prototype);
        obj.__wbg_ptr = ptr;
        CardFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CardFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_card_free(ptr, 0);
    }
    /**
     * Multiply-adds per second in the card's attention, measured for about `ms` milliseconds (a Promise).
     * @param {number} ms
     * @returns {Promise<any>}
     */
    calibrate(ms) {
        const ret = wasm.card_calibrate(this.__wbg_ptr, ms);
        return ret;
    }
    /**
     * `{ working, workingPeak, models }`: bytes of graphics memory in use now.
     * @returns {any}
     */
    memory() {
        const ret = wasm.card_memory(this.__wbg_ptr);
        return ret;
    }
    /**
     * The adapter as the browser describes it.
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.card_name(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Ask the browser for its most capable graphics card and prepare it. `work_mb`: working memory it may use;
     * `models_mb`: graphics memory for fitted models kept between predictions.
     * @param {number} work_mb
     * @param {number} models_mb
     * @returns {Promise<Card>}
     */
    static open(work_mb, models_mb) {
        const ret = wasm.card_open(work_mb, models_mb);
        return ret;
    }
    /**
     * Release everything on the card that is not in use: fitted models, working buffers, weights.
     */
    release() {
        wasm.card_release(this.__wbg_ptr);
    }
    /**
     * Change how much working memory the card may use, in MB.
     * @param {number} mb
     */
    setWorkMb(mb) {
        wasm.card_setWorkMb(this.__wbg_ptr, mb);
    }
    /**
     * True if the browser offers only a program standing in for a card (correct, but slow).
     * @returns {boolean}
     */
    get software() {
        const ret = wasm.card_software(this.__wbg_ptr);
        return ret !== 0;
    }
}
if (Symbol.dispose) Card.prototype[Symbol.dispose] = Card.prototype.free;

/**
 * A training table read from CSV text: features as numbers, the target, and the schema that turns later CSV
 * rows into the same numbers.
 */
export class Table {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TableFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_table_free(ptr, 0);
    }
    /**
     * Number of classes (0 for regression).
     * @returns {number}
     */
    get classes() {
        const ret = wasm.table_classes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get features() {
        const ret = wasm.table_features(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Class labels in class-index order, as a JSON array of strings.
     * @returns {string}
     */
    get labels() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.table_labels(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * How many ensemble members a request for `estimators` really gives on this table (a table with very few
     * columns has fewer distinct arrangements).
     * @param {number} estimators
     * @returns {number}
     */
    members(estimators) {
        const ret = wasm.table_members(this.__wbg_ptr, estimators);
        return ret >>> 0;
    }
    /**
     * Read `csv` (header row first). `regression`: the target is a number to predict; otherwise a class.
     * @param {string} csv
     * @param {string} target
     * @param {boolean} regression
     */
    constructor(csv, target, regression) {
        const ptr0 = passStringToWasm0(csv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(target, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.table_new(ptr0, len0, ptr1, len1, regression);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        TableFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get rows() {
        const ret = wasm.table_rows(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The schema as text: store it with the fitted model and pass it to `featuresFromCsv` when predicting.
     * @returns {string}
     */
    get schema() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.table_schema(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * `rows * features` numbers, row by row, NaN where a cell is missing.
     * @returns {Float64Array}
     */
    get x() {
        const ret = wasm.table_x(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @returns {Uint32Array}
     */
    get yClasses() {
        const ret = wasm.table_yClasses(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float64Array}
     */
    get yNumbers() {
        const ret = wasm.table_yNumbers(this.__wbg_ptr);
        var v1 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) Table.prototype[Symbol.dispose] = Table.prototype.free;

export class Weights {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WeightsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_weights_free(ptr, 0);
    }
    /**
     * `fitClassifierMember` on the graphics card: a Promise of the fitted member's bytes (the same format, so
     * it predicts on either the card or the processor).
     * @param {Card} card
     * @param {Float64Array} x
     * @param {Uint32Array} y
     * @param {number} rows
     * @param {number} features
     * @param {number} estimators
     * @param {number} member
     * @param {string} kv
     * @param {string} schema
     * @returns {Promise<any>}
     */
    fitClassifierMemberCard(card, x, y, rows, features, estimators, member, kv, schema) {
        _assertClass(card, Card);
        const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(y, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(kv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(schema, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.weights_fitClassifierMemberCard(this.__wbg_ptr, card.__wbg_ptr, ptr0, len0, ptr1, len1, rows, features, estimators, member, ptr2, len2, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * Fit ONE member (`member`, counting from 0) of an ensemble of `estimators`. Returns a fitted model holding
     * just that member. `schema` (may be empty) is stored inside it as `table/schema`.
     * @param {Float64Array} x
     * @param {Uint32Array} y
     * @param {number} rows
     * @param {number} features
     * @param {number} estimators
     * @param {number} member
     * @param {string} kv
     * @param {string} schema
     * @returns {Uint8Array}
     */
    fitClassifierMember(x, y, rows, features, estimators, member, kv, schema) {
        const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(y, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(kv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(schema, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.weights_fitClassifierMember(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows, features, estimators, member, ptr2, len2, ptr3, len3);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v5;
    }
    /**
     * Fit a classifier. `x` is `rows * features` numbers row by row (NaN where missing); `y` holds class indices
     * `0..classes`. Returns the fitted model as bytes.
     * @param {Float64Array} x
     * @param {Uint32Array} y
     * @param {number} rows
     * @param {number} features
     * @param {number} estimators
     * @param {string} kv
     * @returns {Uint8Array}
     */
    fitClassifier(x, y, rows, features, estimators, kv) {
        const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(y, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(kv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.weights_fitClassifier(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows, features, estimators, ptr2, len2);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v4 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v4;
    }
    /**
     * `fitRegressorMember` on the graphics card (a Promise of the fitted member's bytes).
     * @param {Card} card
     * @param {Float64Array} x
     * @param {Float64Array} y
     * @param {number} rows
     * @param {number} features
     * @param {number} estimators
     * @param {number} member
     * @param {string} kv
     * @param {string} schema
     * @returns {Promise<any>}
     */
    fitRegressorMemberCard(card, x, y, rows, features, estimators, member, kv, schema) {
        _assertClass(card, Card);
        const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(y, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(kv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(schema, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.weights_fitRegressorMemberCard(this.__wbg_ptr, card.__wbg_ptr, ptr0, len0, ptr1, len1, rows, features, estimators, member, ptr2, len2, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return takeFromExternrefTable0(ret[0]);
    }
    /**
     * As `fitClassifierMember`, for regression.
     * @param {Float64Array} x
     * @param {Float64Array} y
     * @param {number} rows
     * @param {number} features
     * @param {number} estimators
     * @param {number} member
     * @param {string} kv
     * @param {string} schema
     * @returns {Uint8Array}
     */
    fitRegressorMember(x, y, rows, features, estimators, member, kv, schema) {
        const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(y, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(kv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(schema, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.weights_fitRegressorMember(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows, features, estimators, member, ptr2, len2, ptr3, len3);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v5 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v5;
    }
    /**
     * Fit a regression model (needs the regression weights). Returns the fitted model as bytes.
     * @param {Float64Array} x
     * @param {Float64Array} y
     * @param {number} rows
     * @param {number} features
     * @param {number} estimators
     * @param {string} kv
     * @returns {Uint8Array}
     */
    fitRegressor(x, y, rows, features, estimators, kv) {
        const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(y, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(kv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.weights_fitRegressor(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows, features, estimators, ptr2, len2);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v4 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v4;
    }
    /**
     * Forget a fitted member kept on the card (by the `key` given to `predictCard`).
     * @param {Card} card
     * @param {string} key
     */
    forgetCard(card, key) {
        _assertClass(card, Card);
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.weights_forgetCard(this.__wbg_ptr, card.__wbg_ptr, ptr0, len0);
    }
    /**
     * True for the regression checkpoint, false for the classifier.
     * @returns {boolean}
     */
    get isRegression() {
        const ret = wasm.weights_isRegression(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * From the bytes of a weight bundle, or of a published PyTorch checkpoint.
     * @param {Uint8Array} bytes
     */
    constructor(bytes) {
        const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.weights_new(ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WeightsFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * `predictSums` (classifier) or `predictRegression` with the fitted member on the graphics card. `key`
     * names the member, so that it stays on the card for the next prediction. A Promise of
     * `{ out, cardLayers, fallbacks, problem }`: `cardLayers` is how many network layers the card ran (0: the
     * processor answered, for instance because the member does not fit the card's budget), `problem` why the
     * card gave up on any part, if it did.
     * @param {Card} card
     * @param {string} key
     * @param {Uint8Array} fitted
     * @param {Float64Array} x
     * @param {number} rows
     * @returns {Promise<any>}
     */
    predictCard(card, key, fitted, x, rows) {
        _assertClass(card, Card);
        const ptr0 = passStringToWasm0(key, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(fitted, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.weights_predictCard(this.__wbg_ptr, card.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, rows);
        return ret;
    }
    /**
     * Class probabilities, `rows * classes` numbers row by row, for new rows `x`.
     * @param {Uint8Array} fitted
     * @param {Float64Array} x
     * @param {number} rows
     * @returns {Float64Array}
     */
    predictProba(fitted, x, rows) {
        const ptr0 = passArray8ToWasm0(fitted, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.weights_predictProba(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v3;
    }
    /**
     * Regression predictions: four numbers per row (mean, median, 5% and 95% quantiles). For a fitted model
     * holding one member, average these over the members (add in member order, then multiply by 1 / members).
     * @param {Uint8Array} fitted
     * @param {Float64Array} x
     * @param {number} rows
     * @returns {Float64Array}
     */
    predictRegression(fitted, x, rows) {
        const ptr0 = passArray8ToWasm0(fitted, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.weights_predictRegression(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v3;
    }
    /**
     * The score sums (`rows * classes`) of the members held in `fitted`, not yet probabilities. Add the arrays
     * from all members' fitted models together, in member order, and pass the total to `combineProba`.
     * @param {Uint8Array} fitted
     * @param {Float64Array} x
     * @param {number} rows
     * @returns {Float64Array}
     */
    predictSums(fitted, x, rows) {
        const ptr0 = passArray8ToWasm0(fitted, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.weights_predictSums(this.__wbg_ptr, ptr0, len0, ptr1, len1, rows);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v3;
    }
}
if (Symbol.dispose) Weights.prototype[Symbol.dispose] = Weights.prototype.free;

/**
 * How fast this machine runs the inner loop, in multiply-adds per second, measured for about `ms` milliseconds.
 * Needs no weights, so a page can call it at once and use it to estimate how long a fit will take.
 * @param {number} ms
 * @returns {number}
 */
export function calibrate(ms) {
    const ret = wasm.calibrate(ms);
    return ret;
}

/**
 * Probabilities (`rows * classes`) from the members' score sums added together in member order.
 * @param {Float64Array} sums
 * @param {number} classes
 * @param {number} members
 * @param {number} temperature
 * @returns {Float64Array}
 */
export function combineProba(sums, classes, members, temperature) {
    const ptr0 = passArrayF64ToWasm0(sums, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.combineProba(ptr0, len0, classes, members, temperature);
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * The temperature class scores are divided by before they become probabilities.
 * @returns {number}
 */
export function defaultTemperature() {
    const ret = wasm.defaultTemperature();
    return ret;
}

/**
 * Describe the columns of CSV text, as JSON:
 * `{"rows": N, "columns": [{"name", "numeric", "distinct", "missing", "examples": [..]}]}`.
 * @param {string} text
 * @returns {string}
 */
export function describeCsv(text) {
    let deferred3_0;
    let deferred3_1;
    try {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.describeCsv(ptr0, len0);
        var ptr2 = ret[0];
        var len2 = ret[1];
        if (ret[3]) {
            ptr2 = 0; len2 = 0;
            throw takeFromExternrefTable0(ret[2]);
        }
        deferred3_0 = ptr2;
        deferred3_1 = len2;
        return getStringFromWasm0(ptr2, len2);
    } finally {
        wasm.__wbindgen_free(deferred3_0, deferred3_1, 1);
    }
}

/**
 * Features (`rows * columns`, row by row) of new CSV rows, by column name, using a stored schema.
 * @param {string} schema
 * @param {string} csv
 * @returns {Float64Array}
 */
export function featuresFromCsv(schema, csv) {
    const ptr0 = passStringToWasm0(schema, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passStringToWasm0(csv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.featuresFromCsv(ptr0, len0, ptr1, len1);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v3 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v3;
}

/**
 * Bytes of memory this WebAssembly instance holds right now. It only ever grows (up to 4 GB), so after an
 * operation this is the most that operation needed.
 * @returns {number}
 */
export function memoryBytes() {
    const ret = wasm.memoryBytes();
    return ret;
}

/**
 * Turn a checkpoint or weight file into a weight file, optionally half size (16-bit floats).
 * @param {Uint8Array} bytes
 * @param {boolean} half
 * @returns {Uint8Array}
 */
export function toBundle(bytes, half) {
    const ptr0 = passArray8ToWasm0(bytes, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.toBundle(ptr0, len0, half);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayU8FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
    return v2;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_30c8987f7c2ed4e2: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_Window_a2a6c4d665047b14: function(arg0) {
            const ret = arg0.Window;
            return ret;
        },
        __wbg_WorkerGlobalScope_2664448a7c667d67: function(arg0) {
            const ret = arg0.WorkerGlobalScope;
            return ret;
        },
        __wbg___wbindgen_debug_string_4687d8d8c2017d52: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_1f9d30630b8b1d3d: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_null_e343b7d08827ba72: function(arg0) {
            const ret = arg0 === null;
            return ret;
        },
        __wbg___wbindgen_is_string_90b56bc79aad6f6c: function(arg0) {
            const ret = typeof(arg0) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_8865fb403f8fe9d8: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_dcc1a90847f04c41: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_beginComputePass_b9a325184985e6ff: function(arg0, arg1) {
            const ret = arg0.beginComputePass(arg1);
            return ret;
        },
        __wbg_call_187d372bd5fdd4aa: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_card_new: function(arg0) {
            const ret = Card.__wrap(arg0);
            return ret;
        },
        __wbg_configure_1e2c1c9edad07d26: function() { return handleError(function (arg0, arg1) {
            arg0.configure(arg1);
        }, arguments); },
        __wbg_copyBufferToBuffer_9c174b96fb08d551: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferToBuffer(arg1, arg2, arg3, arg4, arg5);
        }, arguments); },
        __wbg_createBindGroupLayout_b1bd63b4e88459d8: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBindGroupLayout(arg1);
            return ret;
        }, arguments); },
        __wbg_createBindGroup_f539b26ca341308f: function(arg0, arg1) {
            const ret = arg0.createBindGroup(arg1);
            return ret;
        },
        __wbg_createBuffer_d800e9b1d41b2ee5: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.createBuffer(arg1);
            return ret;
        }, arguments); },
        __wbg_createCommandEncoder_3352d1ffc36c6fc0: function(arg0, arg1) {
            const ret = arg0.createCommandEncoder(arg1);
            return ret;
        },
        __wbg_createComputePipeline_224fa2618d9948a0: function(arg0, arg1) {
            const ret = arg0.createComputePipeline(arg1);
            return ret;
        },
        __wbg_createPipelineLayout_6eab52c327118937: function(arg0, arg1) {
            const ret = arg0.createPipelineLayout(arg1);
            return ret;
        },
        __wbg_createShaderModule_cefa51336cb288ae: function(arg0, arg1) {
            const ret = arg0.createShaderModule(arg1);
            return ret;
        },
        __wbg_description_83b8a393160021b9: function(arg0, arg1) {
            const ret = arg1.description;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_destroy_637537007d9eaa44: function(arg0) {
            arg0.destroy();
        },
        __wbg_dispatchWorkgroups_56b943172790add0: function(arg0, arg1, arg2, arg3) {
            arg0.dispatchWorkgroups(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_end_82fd0c12ba185009: function(arg0) {
            arg0.end();
        },
        __wbg_error_8c8dff942c7c4f11: function(arg0) {
            const ret = arg0.error;
            return ret;
        },
        __wbg_finish_09ec094c10f41e7b: function(arg0) {
            const ret = arg0.finish();
            return ret;
        },
        __wbg_finish_ec1c191f66a895b1: function(arg0, arg1) {
            const ret = arg0.finish(arg1);
            return ret;
        },
        __wbg_getContext_e567868594d2e0c6: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getMappedRange_fb54c6327b2d8d20: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getMappedRange(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_gpu_afdd4387c7afe5f9: function(arg0) {
            const ret = arg0.gpu;
            return ret;
        },
        __wbg_info_971d8b9db3dae69f: function(arg0) {
            const ret = arg0.info;
            return ret;
        },
        __wbg_instanceof_GpuOutOfMemoryError_5c3b8a2499f59adb: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUOutOfMemoryError;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_GpuValidationError_7659e07b12d4e184: function(arg0) {
            let result;
            try {
                result = arg0 instanceof GPUValidationError;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_isFallbackAdapter_4c8cc3b18677460a: function(arg0) {
            const ret = arg0.isFallbackAdapter;
            return ret;
        },
        __wbg_label_7add8cb37a6ef98f: function(arg0, arg1) {
            const ret = arg1.label;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_length_7f3c00c40364105e: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_limits_601ad2e086ef8141: function(arg0) {
            const ret = arg0.limits;
            return ret;
        },
        __wbg_lost_bb454eb62101af1a: function(arg0) {
            const ret = arg0.lost;
            return ret;
        },
        __wbg_mapAsync_b0597127f5037286: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.mapAsync(arg1 >>> 0, arg2, arg3);
            return ret;
        },
        __wbg_maxBindGroupsPlusVertexBuffers_52369f089736ef9d: function(arg0) {
            const ret = arg0.maxBindGroupsPlusVertexBuffers;
            return ret;
        },
        __wbg_maxBindGroups_4e424afe6ce86ca2: function(arg0) {
            const ret = arg0.maxBindGroups;
            return ret;
        },
        __wbg_maxBindingsPerBindGroup_7d035da36821c44f: function(arg0) {
            const ret = arg0.maxBindingsPerBindGroup;
            return ret;
        },
        __wbg_maxBufferSize_423f4a084e32a195: function(arg0) {
            const ret = arg0.maxBufferSize;
            return ret;
        },
        __wbg_maxColorAttachmentBytesPerSample_c4cd9126f6d287c6: function(arg0) {
            const ret = arg0.maxColorAttachmentBytesPerSample;
            return ret;
        },
        __wbg_maxColorAttachments_d924670762b9e250: function(arg0) {
            const ret = arg0.maxColorAttachments;
            return ret;
        },
        __wbg_maxComputeInvocationsPerWorkgroup_707a3868f7cebb59: function(arg0) {
            const ret = arg0.maxComputeInvocationsPerWorkgroup;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeX_0a4d99463cbd6e5e: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeX;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeY_85123ea0587f7558: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeY;
            return ret;
        },
        __wbg_maxComputeWorkgroupSizeZ_a3186b4c5267d44f: function(arg0) {
            const ret = arg0.maxComputeWorkgroupSizeZ;
            return ret;
        },
        __wbg_maxComputeWorkgroupStorageSize_57b297355cfb6204: function(arg0) {
            const ret = arg0.maxComputeWorkgroupStorageSize;
            return ret;
        },
        __wbg_maxComputeWorkgroupsPerDimension_4158f95e673d54c4: function(arg0) {
            const ret = arg0.maxComputeWorkgroupsPerDimension;
            return ret;
        },
        __wbg_maxDynamicStorageBuffersPerPipelineLayout_226b0b70910aa16c: function(arg0) {
            const ret = arg0.maxDynamicStorageBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxDynamicUniformBuffersPerPipelineLayout_0e835fda711fc7e6: function(arg0) {
            const ret = arg0.maxDynamicUniformBuffersPerPipelineLayout;
            return ret;
        },
        __wbg_maxInterStageShaderVariables_8c4a1d727e2aa35a: function(arg0) {
            const ret = arg0.maxInterStageShaderVariables;
            return ret;
        },
        __wbg_maxSampledTexturesPerShaderStage_6675f5e91d9a728a: function(arg0) {
            const ret = arg0.maxSampledTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxSamplersPerShaderStage_1910fa38a6ed1e1f: function(arg0) {
            const ret = arg0.maxSamplersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageBufferBindingSize_2e244bded070b18d: function(arg0) {
            const ret = arg0.maxStorageBufferBindingSize;
            return ret;
        },
        __wbg_maxStorageBuffersPerShaderStage_a285f3ebca51ca0d: function(arg0) {
            const ret = arg0.maxStorageBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxStorageTexturesPerShaderStage_7aa946f0fc322a2b: function(arg0) {
            const ret = arg0.maxStorageTexturesPerShaderStage;
            return ret;
        },
        __wbg_maxTextureArrayLayers_0e699147ad00502d: function(arg0) {
            const ret = arg0.maxTextureArrayLayers;
            return ret;
        },
        __wbg_maxTextureDimension1D_aabf6add54decfe2: function(arg0) {
            const ret = arg0.maxTextureDimension1D;
            return ret;
        },
        __wbg_maxTextureDimension2D_dd598b27e9c0c1c4: function(arg0) {
            const ret = arg0.maxTextureDimension2D;
            return ret;
        },
        __wbg_maxTextureDimension3D_f944266c65dfd1a9: function(arg0) {
            const ret = arg0.maxTextureDimension3D;
            return ret;
        },
        __wbg_maxUniformBufferBindingSize_59fa6be7cfbeeb53: function(arg0) {
            const ret = arg0.maxUniformBufferBindingSize;
            return ret;
        },
        __wbg_maxUniformBuffersPerShaderStage_bee5f00a4d706c7f: function(arg0) {
            const ret = arg0.maxUniformBuffersPerShaderStage;
            return ret;
        },
        __wbg_maxVertexAttributes_5cf6392c4e9033fe: function(arg0) {
            const ret = arg0.maxVertexAttributes;
            return ret;
        },
        __wbg_maxVertexBufferArrayStride_548baa887375d865: function(arg0) {
            const ret = arg0.maxVertexBufferArrayStride;
            return ret;
        },
        __wbg_maxVertexBuffers_75d881156591f5da: function(arg0) {
            const ret = arg0.maxVertexBuffers;
            return ret;
        },
        __wbg_message_2aad50368e15e8f1: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_message_ab4cad9eeb386607: function(arg0, arg1) {
            const ret = arg1.message;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_minStorageBufferOffsetAlignment_5ba9b77792bdadb3: function(arg0) {
            const ret = arg0.minStorageBufferOffsetAlignment;
            return ret;
        },
        __wbg_minUniformBufferOffsetAlignment_ab7d52a5293b22bd: function(arg0) {
            const ret = arg0.minUniformBufferOffsetAlignment;
            return ret;
        },
        __wbg_navigator_2156486643462a87: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_navigator_c9bce48d4b578c9c: function(arg0) {
            const ret = arg0.navigator;
            return ret;
        },
        __wbg_new_617a8cdb8bb1130e: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_a217df4351db4b9b: function() { return handleError(function (arg0, arg1) {
            const ret = new OffscreenCanvas(arg0 >>> 0, arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_new_from_slice_9a868026ffa4208a: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_from_slice_f545fd22ddc142b8: function(arg0, arg1) {
            const ret = new Float64Array(getArrayF64FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_typed_689a3a281a8d4da6: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_typed_b01cb72a8af741a3: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___js_sys_89cb31567df83114___Function_fn_wasm_bindgen_7112a48bcae1756b___JsValue_____wasm_bindgen_7112a48bcae1756b___sys__Undefined___js_sys_89cb31567df83114___Function_fn_wasm_bindgen_7112a48bcae1756b___JsValue_____wasm_bindgen_7112a48bcae1756b___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_with_byte_offset_and_length_2f5d7fc2a828b74d: function(arg0, arg1, arg2) {
            const ret = new Uint8Array(arg0, arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_now_e7c6795a7f81e10f: function(arg0) {
            const ret = arg0.now();
            return ret;
        },
        __wbg_onSubmittedWorkDone_1190213cee1ecf7e: function(arg0) {
            const ret = arg0.onSubmittedWorkDone();
            return ret;
        },
        __wbg_performance_3fcf6e32a7e1ed0a: function(arg0) {
            const ret = arg0.performance;
            return ret;
        },
        __wbg_prototypesetcall_bc27214492979395: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_queueMicrotask_9833f9a49df95a49: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_a72f977e97f23c5f: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_queue_7b62c28143d44293: function(arg0) {
            const ret = arg0.queue;
            return ret;
        },
        __wbg_reason_b4fa3af867fc4430: function(arg0) {
            const ret = arg0.reason;
            return (__wbindgen_enum_GpuDeviceLostReason.indexOf(ret) + 1 || 3) - 1;
        },
        __wbg_requestAdapter_a539af006419f2e9: function(arg0, arg1) {
            const ret = arg0.requestAdapter(arg1);
            return ret;
        },
        __wbg_requestDevice_5cb8a582e55d08cb: function(arg0, arg1) {
            const ret = arg0.requestDevice(arg1);
            return ret;
        },
        __wbg_resolve_0076e10020304ede: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_setBindGroup_c83391351ce68826: function(arg0, arg1, arg2) {
            arg0.setBindGroup(arg1 >>> 0, arg2);
        },
        __wbg_setPipeline_5146f7b8d2b4af5c: function(arg0, arg1) {
            arg0.setPipeline(arg1);
        },
        __wbg_set_145a351398b48c65: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_34d08fd992d43d61: function(arg0, arg1, arg2) {
            arg0.set(arg1, arg2 >>> 0);
        },
        __wbg_set_access_a099cfbbeec9b96f: function(arg0, arg1) {
            arg0.access = __wbindgen_enum_GpuStorageTextureAccess[arg1];
        },
        __wbg_set_beginning_of_pass_write_index_fa9ae10d5d2804ce: function(arg0, arg1) {
            arg0.beginningOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_bind_group_layouts_458c44ba55100b82: function(arg0, arg1, arg2) {
            arg0.bindGroupLayouts = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_binding_81b3fac7f7acaf8d: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_binding_b6cee57f35ac5190: function(arg0, arg1) {
            arg0.binding = arg1 >>> 0;
        },
        __wbg_set_buffer_1548ae88a9188037: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_buffer_8d0ac64ad20dfc84: function(arg0, arg1) {
            arg0.buffer = arg1;
        },
        __wbg_set_code_5d5b0b9e2fd0dca7: function(arg0, arg1, arg2) {
            arg0.code = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_compute_bc754d9e37f6e4c9: function(arg0, arg1) {
            arg0.compute = arg1;
        },
        __wbg_set_device_210484a77b675c9c: function(arg0, arg1) {
            arg0.device = arg1;
        },
        __wbg_set_end_of_pass_write_index_b54a8e3802b9ae0d: function(arg0, arg1) {
            arg0.endOfPassWriteIndex = arg1 >>> 0;
        },
        __wbg_set_entries_6f866302103b81e9: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entries_f26b77ab9548e906: function(arg0, arg1, arg2) {
            arg0.entries = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_entry_point_ac8db535971761fe: function(arg0, arg1, arg2) {
            arg0.entryPoint = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_external_texture_7f966c604c4f8098: function(arg0, arg1) {
            arg0.externalTexture = arg1;
        },
        __wbg_set_format_92732ea75d3b79f5: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_format_f009e603f7d4c28e: function(arg0, arg1) {
            arg0.format = __wbindgen_enum_GpuTextureFormat[arg1];
        },
        __wbg_set_has_dynamic_offset_0c72ffa900c5a269: function(arg0, arg1) {
            arg0.hasDynamicOffset = arg1 !== 0;
        },
        __wbg_set_label_09e2da5f8522dbe8: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_17202740051e9722: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_17b4858e1eef23dd: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_2fefb39c0e0dbbe8: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_612add98a4398f92: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_92cd3811e96b487c: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_c7987704d29f284b: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_label_ee172cd5f6a96961: function(arg0, arg1, arg2) {
            arg0.label = getStringFromWasm0(arg1, arg2);
        },
        __wbg_set_layout_454e3a091b390cd4: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_4b5d5b12fbb72a76: function(arg0, arg1) {
            arg0.layout = arg1;
        },
        __wbg_set_layout_gpu_auto_layout_mode_ee432e515c357fa8: function(arg0, arg1) {
            arg0.layout = __wbindgen_enum_GpuAutoLayoutMode[arg1];
        },
        __wbg_set_mapped_at_creation_3f320fef6761b02c: function(arg0, arg1) {
            arg0.mappedAtCreation = arg1 !== 0;
        },
        __wbg_set_min_binding_size_f64_897e3cd4496ddec9: function(arg0, arg1) {
            arg0.minBindingSize = arg1;
        },
        __wbg_set_module_c9946af218d23b53: function(arg0, arg1) {
            arg0.module = arg1;
        },
        __wbg_set_multisampled_039f032dc4b67367: function(arg0, arg1) {
            arg0.multisampled = arg1 !== 0;
        },
        __wbg_set_offset_f64_127e8a0aa5c5485a: function(arg0, arg1) {
            arg0.offset = arg1;
        },
        __wbg_set_onuncapturederror_bb813f1102da2296: function(arg0, arg1) {
            arg0.onuncapturederror = arg1;
        },
        __wbg_set_power_preference_b42d00a8facfbade: function(arg0, arg1) {
            arg0.powerPreference = __wbindgen_enum_GpuPowerPreference[arg1];
        },
        __wbg_set_query_set_f9a94586851fdba2: function(arg0, arg1) {
            arg0.querySet = arg1;
        },
        __wbg_set_required_features_bbab71414c45e621: function(arg0, arg1, arg2) {
            arg0.requiredFeatures = getArrayJsValueViewFromWasm0(arg1, arg2);
        },
        __wbg_set_required_limits_837f62d865e7cfac: function(arg0, arg1) {
            arg0.requiredLimits = arg1;
        },
        __wbg_set_resource_8fd8658b30d86ecf: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_buffer_binding_33099b25da65b610: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_resource_gpu_texture_view_4cffe7bc7c8e5cbe: function(arg0, arg1) {
            arg0.resource = arg1;
        },
        __wbg_set_sample_type_ebc5fcd029513bda: function(arg0, arg1) {
            arg0.sampleType = __wbindgen_enum_GpuTextureSampleType[arg1];
        },
        __wbg_set_sampler_89cb4a7efcfc6005: function(arg0, arg1) {
            arg0.sampler = arg1;
        },
        __wbg_set_size_f64_2f591b0654540477: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_size_f64_e844c985b8f95261: function(arg0, arg1) {
            arg0.size = arg1;
        },
        __wbg_set_storage_texture_786aea7c5773b6c1: function(arg0, arg1) {
            arg0.storageTexture = arg1;
        },
        __wbg_set_texture_a33be3fe02ac6264: function(arg0, arg1) {
            arg0.texture = arg1;
        },
        __wbg_set_timestamp_writes_5f240bbaad97fcaf: function(arg0, arg1) {
            arg0.timestampWrites = arg1;
        },
        __wbg_set_type_43e0092f16775979: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuSamplerBindingType[arg1];
        },
        __wbg_set_type_79cec55caf4cdb6d: function(arg0, arg1) {
            arg0.type = __wbindgen_enum_GpuBufferBindingType[arg1];
        },
        __wbg_set_usage_f3e34822998d2147: function(arg0, arg1) {
            arg0.usage = arg1 >>> 0;
        },
        __wbg_set_view_dimension_893e2d16561e56e8: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_view_dimension_f2c5fe4bf927c3fe: function(arg0, arg1) {
            arg0.viewDimension = __wbindgen_enum_GpuTextureViewDimension[arg1];
        },
        __wbg_set_visibility_d8a6821789538c25: function(arg0, arg1) {
            arg0.visibility = arg1 >>> 0;
        },
        __wbg_static_accessor_GLOBAL_266715b9d96ba635: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_10fb7dc1ae063179: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_0b583911f537483a: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_d7f903d1508cbdc4: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subgroupMaxSize_b43be0aa16182403: function(arg0) {
            const ret = arg0.subgroupMaxSize;
            return ret;
        },
        __wbg_subgroupMinSize_03feb6ee0cda6775: function(arg0) {
            const ret = arg0.subgroupMinSize;
            return ret;
        },
        __wbg_submit_077c85cc28e36892: function(arg0, arg1, arg2) {
            arg0.submit(getArrayJsValueViewFromWasm0(arg1, arg2));
        },
        __wbg_then_c8a35d4ad59c6b8e: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_then_c949d5a25a4e78f8: function(arg0, arg1, arg2) {
            const ret = arg0.then(arg1, arg2);
            return ret;
        },
        __wbg_then_e71170d78fcf8954: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_unconfigure_835307f58dc68d80: function(arg0) {
            arg0.unconfigure();
        },
        __wbg_unmap_6a96b14c9ef5f7f5: function(arg0) {
            arg0.unmap();
        },
        __wbg_valueOf_02fd4d101b7553c7: function(arg0) {
            const ret = arg0.valueOf();
            return ret;
        },
        __wbg_writeBuffer_f4bb3f54adfe1330: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.writeBuffer(arg1, arg2, getArrayU8FromWasm0(arg3, arg4), arg5, arg6);
        }, arguments); },
        __wbindgen_generic_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 77, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true_);
            return ret;
        },
        __wbindgen_generic_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUDevice")], shim_idx: 77, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__39);
            return ret;
        },
        __wbindgen_generic_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUDeviceLostInfo")], shim_idx: 85, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true_);
            return ret;
        },
        __wbindgen_generic_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("GPUUncapturedErrorEvent")], shim_idx: 85, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true__41);
            return ret;
        },
        __wbindgen_generic_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("any")], shim_idx: 77, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__42);
            return ret;
        },
        __wbindgen_generic_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("undefined")], shim_idx: 77, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__43);
            return ret;
        },
        __wbindgen_generic_0000000000000007: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_generic_0000000000000008: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_generic_0000000000000009: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./tabicl_anywhere_wasm_bg.js": import0,
    };
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true_(arg0, arg1, arg2);
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true__41(arg0, arg1, arg2) {
    wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuDeviceLostInfo__GpuDeviceLostInfo______true__41(arg0, arg1, arg2);
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__39(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__39(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__42(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__42(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__43(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___wasm_bindgen_7112a48bcae1756b___sys__JsNullable_wgpu_60fc19ae4b8c5396___backend__webgpu__webgpu_sys__gen_GpuAdapter__GpuAdapter___core_ed718c3d60ebd546___result__Result_____wasm_bindgen_7112a48bcae1756b___JsError___true__43(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___js_sys_89cb31567df83114___Function_fn_wasm_bindgen_7112a48bcae1756b___JsValue_____wasm_bindgen_7112a48bcae1756b___sys__Undefined___js_sys_89cb31567df83114___Function_fn_wasm_bindgen_7112a48bcae1756b___JsValue_____wasm_bindgen_7112a48bcae1756b___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_7112a48bcae1756b___convert__closures_____invoke___js_sys_89cb31567df83114___Function_fn_wasm_bindgen_7112a48bcae1756b___JsValue_____wasm_bindgen_7112a48bcae1756b___sys__Undefined___js_sys_89cb31567df83114___Function_fn_wasm_bindgen_7112a48bcae1756b___JsValue_____wasm_bindgen_7112a48bcae1756b___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}


const __wbindgen_enum_GpuAutoLayoutMode = ["auto"];


const __wbindgen_enum_GpuBufferBindingType = ["uniform", "storage", "read-only-storage"];


const __wbindgen_enum_GpuDeviceLostReason = ["unknown", "destroyed"];


const __wbindgen_enum_GpuPowerPreference = ["low-power", "high-performance"];


const __wbindgen_enum_GpuSamplerBindingType = ["filtering", "non-filtering", "comparison"];


const __wbindgen_enum_GpuStorageTextureAccess = ["write-only", "read-only", "read-write"];


const __wbindgen_enum_GpuTextureFormat = ["r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint", "r32float", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint", "rg32float", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float", "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"];


const __wbindgen_enum_GpuTextureSampleType = ["float", "unfilterable-float", "depth", "sint", "uint"];


const __wbindgen_enum_GpuTextureViewDimension = ["1d", "2d", "2d-array", "cube", "cube-array", "3d"];
const CardFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_card_free(ptr, 1));
const TableFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_table_free(ptr, 1));
const WeightsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_weights_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayJsValueViewFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    const mem = getDataViewMemory0();
    const result = [];
    for (let i = ptr; i < ptr + 4 * len; i += 4) {
        result.push(wasm.__wbindgen_externrefs.get(mem.getUint32(i, true)));
    }
    return result;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('tabicl_anywhere_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
