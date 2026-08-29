// static/js/extract-fields.js
// Offline, in-browser port of core/ai_utils.py's regex-based document type
// detection and structured field extraction. No network calls — runs
// entirely on OCR text produced by tesseract-ocr.js.

(function (global) {
    'use strict';

    const MIN_TEXT_LENGTH_FOR_DETECTION = 20;
    const MIN_TEXT_LENGTH_FOR_EXTRACTION = 30;

    // ============================================================
    //  PATTERNS  (mirrors core/ai_utils.py)
    // ============================================================
    const PATTERNS = {
        AADHAAR: /\b\d{4}\s?\d{4}\s?\d{4}\b/,
        PAN: /\b[A-Z]{5}[0-9]{4}[A-Z]\b/,
        VOTER_ID: /\b[A-Z]{3}[0-9]{7}\b/,
        PASSPORT: /\b[A-Z][0-9]{7}\b/,
        DL: /\b[A-Z]{2}\d{2}\s?\d{11}\b/,
        VEHICLE_REG: /\b[A-Z]{2}[0-9]{2}[A-Z]{4}[0-9]{4}\b/,

        CERTIFICATE_NUMBER: /(?:certificate|cert|no|number)\s*[:#]?\s*([A-Z0-9\-/]{6,20})/i,
        REGISTRATION_NUMBER: /(?:registration|reg|serial)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]{6,20})/i,

        SURVEY_NUMBER: /(?:survey|s\.?no\.?|s\.no)\s*[:#]?\s*([0-9/\-]+)/i,
        PLOT_NUMBER: /(?:plot|p\.?no\.?|p\.no)\s*[:#]?\s*([0-9/\-]+)/i,

        SEAT_NUMBER: /(?:seat|roll|registration)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)/i,
        PERCENTAGE: /(\d{1,3}\.?\d{1,2})%/i,

        CASTE: /(?:caste|category|sub-caste)\s*[:]?\s*([A-Za-z ]+)/i,

        GST: /\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ\d\b/,
        EMAIL: /[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+/,
        PHONE: /\b[6-9]\d{9}\b/,
        DATE: /\b\d{2}[-/.]\d{2}[-/.]\d{4}\b/,
        YEAR: /\b(19|20)\d{2}\b/,

        NAME: /(?:full\s*name|name|applicant|student|owner)\s*[:]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)/i,
        NAME_FALLBACK: /\n([A-Z][A-Za-z ]{3,})\n/,

        GENDER: /\b(MALE|FEMALE|Male|Female|M|F)\b/,
        ADDRESS: /(?:address|residence|permanent|present)\s*[:]?\s*(.*?)(?=\n|$)/i,
        NATIONALITY: /nationality\s*[:]?\s*([A-Za-z ]+)/i,
        FATHER_NAME: /(?:father|husband|spouse)\s*['\u2019]?s?\s*(?:name)?\s*[:]?\s*([A-Za-z ]+)/i,
        MOTHER_NAME: /mother\s*['\u2019]?s?\s*(?:name)?\s*[:]?\s*([A-Za-z ]+)/i,

        INVOICE: /invoice\s*(?:no|number)?\s*[:]?\s*([A-Za-z0-9\-/]+)/i,
        TOTAL: /(?:total\s*amount|grand\s*total|amount\s*paid|total)\s*[:]?\s*[\u20B9$]?\s*([\d,]+\.\d{2})/i,

        RATION_CARD: /(?:ration|rc)\s*(?:card)?\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)/i,
        ENGINE_NUMBER: /engine\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)/i,
        CHASSIS_NUMBER: /(?:chassis|frame|vin)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)/i,
        BLOOD_GROUP: /blood\s*(?:group|type)?\s*[:]?\s*([A-Za-z+-]+)/i,
        DISTRICT: /(?:district|constituency|taluka|tehsil)\s*[:]?\s*([A-Za-z ]+)/i,
        VILLAGE: /(?:village|town|city|taluka)\s*[:]?\s*([A-Za-z ]+)/i,
        INCOME: /(?:income|annual|yearly)\s*[:]?\s*[\u20B9$]?\s*([\d,]+\.?\d*)/i,
        VALIDITY: /(?:valid|expiry|expiration|validity)\s*(?:up to|till|date)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i,
        ISSUE_DATE: /(?:issue|issued|date of issue)\s*(?:date)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i,
    };

    // ============================================================
    //  TEXT CLEANING
    // ============================================================
    function cleanOcrText(text) {
        if (!text || typeof text !== 'string') return '';
        // eslint-disable-next-line no-control-regex
        text = text.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
        text = text.replace(/\n\s*\n/g, '\n');
        text = text.replace(/[ \t]+/g, ' ');
        return text.trim();
    }

    function maskSensitiveData(text) {
        if (!text) return '';
        text = text.replace(new RegExp(PATTERNS.AADHAAR, 'g'), 'XXXX XXXX XXXX');
        text = text.replace(new RegExp(PATTERNS.PAN, 'g'), 'XXXXX0000X');
        text = text.replace(new RegExp(PATTERNS.VOTER_ID, 'g'), 'XXX0000000');
        text = text.replace(new RegExp(PATTERNS.PASSPORT, 'g'), 'X0000000');
        return text;
    }

    function extractField(text, pattern, group, defaultVal) {
        group = group === undefined ? 1 : group;
        defaultVal = defaultVal === undefined ? '' : defaultVal;
        const match = text.match(pattern);
        if (match && match[group] !== undefined) {
            return match[group].trim();
        }
        return defaultVal;
    }

    function nameOrFallback(text) {
        const m = text.match(PATTERNS.NAME);
        if (m) return m[1].trim();
        const f = text.match(PATTERNS.NAME_FALLBACK);
        if (f) return f[1].trim();
        return null;
    }

    // ============================================================
    //  DOCUMENT TYPE DETECTION  (mirrors detect_document_type)
    // ============================================================
    function detectDocumentType(text) {
        if (!text || typeof text !== 'string') return null;
        const cleaned = cleanOcrText(text);
        if (cleaned.length < MIN_TEXT_LENGTH_FOR_DETECTION) return null;
        const t = cleaned.toLowerCase();

        if (t.includes('aadhaar') || t.includes('uidai')) return 'aadhaar_card';
        if ((t.includes('income tax') && t.includes('permanent account')) ||
            (t.includes('pan') && t.includes('department'))) return 'pan_card';
        if (t.includes('voter id') || t.includes('election') || t.includes('epic')) return 'voter_id_card';
        if (t.includes('passport') && t.includes('india')) return 'passport';
        if (t.includes('driving') && (t.includes('licence') || t.includes('license'))) return 'driving_license';
        if (t.includes('registration certificate') || t.includes('rc') ||
            (t.includes('vehicle') && t.includes('register'))) return 'vehicle_registration_certificate';

        if (t.includes('domicile') || t.includes('resident')) return 'domicile_certificate';
        if (t.includes('nationality') || t.includes('citizenship')) return 'nationality_certificate';
        if (t.includes('birth') && t.includes('certificate')) return 'birth_certificate';
        if (t.includes('marriage') && t.includes('certificate')) return 'marriage_certificate';
        if (t.includes('death') && t.includes('certificate')) return 'death_certificate';

        if (t.includes('property card') || t.includes('7/12') || t.includes('8a')) return 'property_card';
        if (t.includes('income certificate') || t.includes('income proof')) return 'income_certificate';
        if (t.includes('ration card') || t.includes('ration')) return 'ration_card';

        if (t.includes('school leaving') || t.includes(' lc ') || t.includes('transfer certificate')) return 'school_leaving_certificate';
        if (t.includes('ssc') && t.includes('marksheet')) return 'ssc_marksheet';
        if (t.includes('hsc') && t.includes('marksheet')) return 'hsc_marksheet';
        if (t.includes('degree') && t.includes('certificate')) return 'degree_certificate';
        if (t.includes('board') && t.includes('passing')) return 'board_passing_certificate';

        if (t.includes('caste') && t.includes('certificate')) {
            return t.includes('validity') ? 'caste_validity_certificate' : 'caste_certificate';
        }
        if (t.includes('non creamy') || t.includes('ncl')) return 'non_creamy_layer_certificate';
        if (t.includes('ews') || t.includes('economically weaker')) return 'ews_certificate';

        if (t.includes('gst') && t.includes('certificate')) return 'gst_certificate';
        if (t.includes('invoice') || t.includes('bill')) return 'invoice';

        return 'other_document';
    }

    // ============================================================
    //  EXTRACTORS  (mirrors the _extract_* functions)
    // ============================================================
    const EXTRACTORS = {
        aadhaar_card(text) {
            const data = {};
            const aadhaar = text.match(PATTERNS.AADHAAR);
            if (aadhaar) data['Aadhaar Number'] = aadhaar[0];
            const name = nameOrFallback(text);
            if (name) data['Full Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const gender = text.match(PATTERNS.GENDER);
            if (gender) data['Gender'] = gender[0];
            const addr = text.match(PATTERNS.ADDRESS);
            if (addr) data['Address'] = addr[1].trim();
            if (text.toLowerCase().includes('qr')) data['QR Code'] = 'Present';
            return data;
        },
        pan_card(text) {
            const data = {};
            const pan = text.match(PATTERNS.PAN);
            if (pan) data['PAN Number'] = pan[0];
            const name = nameOrFallback(text);
            if (name) data['Full Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const father = text.match(PATTERNS.FATHER_NAME);
            if (father) data["Father's Name"] = father[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            return data;
        },
        voter_id_card(text) {
            const data = {};
            const voterId = text.match(PATTERNS.VOTER_ID);
            if (voterId) data['Voter ID Number'] = voterId[0];
            const name = nameOrFallback(text);
            if (name) data['Full Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const gender = text.match(PATTERNS.GENDER);
            if (gender) data['Gender'] = gender[0];
            const father = text.match(PATTERNS.FATHER_NAME);
            if (father) data['Father/Husband Name'] = father[1].trim();
            const district = text.match(PATTERNS.DISTRICT);
            if (district) data['Assembly Constituency'] = district[1].trim();
            const addr = text.match(PATTERNS.ADDRESS);
            if (addr) data['Address'] = addr[1].trim();
            return data;
        },
        passport(text) {
            const data = {};
            const passport = text.match(PATTERNS.PASSPORT);
            if (passport) data['Passport Number'] = passport[0];
            const name = nameOrFallback(text);
            if (name) data['Full Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const place = extractField(text, /place of birth\s*[:]?\s*([A-Za-z ]+)/i);
            if (place) data['Place of Birth'] = place;
            const nationality = text.match(PATTERNS.NATIONALITY);
            if (nationality) data['Nationality'] = nationality[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            const validity = text.match(PATTERNS.VALIDITY);
            if (validity) data['Expiry Date'] = validity[1].trim();
            const fileNo = extractField(text, /file\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)/i);
            if (fileNo) data['File Number'] = fileNo;
            return data;
        },
        driving_license(text) {
            const data = {};
            const dl = text.match(PATTERNS.DL);
            if (dl) data['License Number'] = dl[0];
            const name = nameOrFallback(text);
            if (name) data['Full Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const blood = text.match(PATTERNS.BLOOD_GROUP);
            if (blood) data['Blood Group'] = blood[1].trim();
            const addr = text.match(PATTERNS.ADDRESS);
            if (addr) data['Address'] = addr[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            const validity = text.match(PATTERNS.VALIDITY);
            if (validity) data['Valid Until'] = validity[1].trim();
            const vClass = extractField(text, /(?:vehicle|class)\s*(?:type)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (vClass) data['Vehicle Class'] = vClass;
            return data;
        },
        vehicle_registration_certificate(text) {
            const data = {};
            const reg = text.match(PATTERNS.VEHICLE_REG);
            if (reg) data['Registration Number'] = reg[0];
            const name = nameOrFallback(text);
            if (name) data['Owner Name'] = name;
            const vType = extractField(text, /(?:vehicle|type|model)\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (vType) data['Vehicle Type'] = vType;
            const regDate = text.match(PATTERNS.DATE);
            if (regDate) data['Registration Date'] = regDate[0];
            const engine = text.match(PATTERNS.ENGINE_NUMBER);
            if (engine) data['Engine Number'] = engine[1].trim();
            const chassis = text.match(PATTERNS.CHASSIS_NUMBER);
            if (chassis) data['Chassis Number'] = chassis[1].trim();
            const manufacturer = extractField(text, /(?:manufacturer|make)\s*[:]?\s*([A-Za-z ]+)/i);
            if (manufacturer) data['Vehicle Manufacturer'] = manufacturer;
            const fuel = extractField(text, /fuel\s*(?:type)?\s*[:]?\s*([A-Za-z ]+)/i);
            if (fuel) data['Fuel Type'] = fuel;
            const color = extractField(text, /color\s*[:]?\s*([A-Za-z ]+)/i);
            if (color) data['Color'] = color;
            const validity = text.match(PATTERNS.VALIDITY);
            if (validity) data['Valid Until'] = validity[1].trim();
            return data;
        },
        domicile_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const period = extractField(text, /(?:period|duration)\s*(?:of residence)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (period) data['Period of Residence'] = period;
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Certificate Number'] = cert[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            const authority = extractField(text, /(?:authority|issuing|by)\s*[:]?\s*([A-Za-z ]+)/i);
            if (authority) data['Authority'] = authority;
            return data;
        },
        nationality_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const nationality = text.match(PATTERNS.NATIONALITY);
            if (nationality) data['Citizenship Declaration'] = nationality[1].trim();
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Outward Number'] = cert[1].trim();
            const authority = extractField(text, /(?:authority|issuing|by)\s*[:]?\s*([A-Za-z ]+)/i);
            if (authority) data['Issuing Authority'] = authority;
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            return data;
        },
        birth_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Full Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const place = extractField(text, /place of birth\s*[:]?\s*([A-Za-z ]+)/i);
            if (place) data['Place of Birth'] = place;
            const mother = text.match(PATTERNS.MOTHER_NAME);
            if (mother) data["Mother's Name"] = mother[1].trim();
            const father = text.match(PATTERNS.FATHER_NAME);
            if (father) data["Father's Name"] = father[1].trim();
            const reg = text.match(PATTERNS.REGISTRATION_NUMBER);
            if (reg) data['Registration Number'] = reg[1].trim();
            const regDate = extractField(text, /date of registration\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (regDate) data['Date of Registration'] = regDate;
            return data;
        },
        marriage_certificate(text) {
            const data = {};
            const husband = text.match(PATTERNS.FATHER_NAME);
            if (husband) data["Husband's Name"] = husband[1].trim();
            const wife = extractField(text, /wife\s*['\u2019]?s?\s*(?:name)?\s*[:]?\s*([A-Za-z ]+)/i);
            if (wife) data["Wife's Name"] = wife;
            const mdate = text.match(PATTERNS.DATE);
            if (mdate) data['Date of Marriage'] = mdate[0];
            const place = extractField(text, /place of marriage\s*[:]?\s*([A-Za-z ]+)/i);
            if (place) data['Place of Marriage'] = place;
            const reg = text.match(PATTERNS.REGISTRATION_NUMBER);
            if (reg) data['Registration Number'] = reg[1].trim();
            const regDate = extractField(text, /date of registration\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (regDate) data['Date of Registration'] = regDate;
            return data;
        },
        death_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Deceased Name'] = name;
            const ddate = text.match(PATTERNS.DATE);
            if (ddate) data['Date of Death'] = ddate[0];
            const place = extractField(text, /place of death\s*[:]?\s*([A-Za-z ]+)/i);
            if (place) data['Place of Death'] = place;
            const cause = extractField(text, /cause of death\s*[:]?\s*([A-Za-z ]+)/i);
            if (cause) data['Cause of Death'] = cause;
            const reg = text.match(PATTERNS.REGISTRATION_NUMBER);
            if (reg) data['Registration Number'] = reg[1].trim();
            const regDate = extractField(text, /date of registration\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (regDate) data['Date of Registration'] = regDate;
            return data;
        },
        property_card(text) {
            const data = {};
            const survey = text.match(PATTERNS.SURVEY_NUMBER);
            if (survey) data['Survey Number'] = survey[1].trim();
            const plot = text.match(PATTERNS.PLOT_NUMBER);
            if (plot) data['Plot Number'] = plot[1].trim();
            const name = nameOrFallback(text);
            if (name) data['Owner Name'] = name;
            const area = extractField(text, /(?:area|land)\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (area) data['Land Area'] = area;
            const village = text.match(PATTERNS.VILLAGE);
            if (village) data['Village Name'] = village[1].trim();
            const tax = extractField(text, /(?:tax|assessment)\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (tax) data['Tax Assessment Details'] = tax;
            return data;
        },
        income_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const income = text.match(PATTERNS.INCOME);
            if (income) data['Annual Family Income'] = income[1].trim();
            const year = text.match(PATTERNS.YEAR);
            if (year) data['Financial Year'] = year[0];
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Certificate Number'] = cert[1].trim();
            const authority = extractField(text, /(?:authority|issuing|by)\s*[:]?\s*([A-Za-z ]+)/i);
            if (authority) data['Authority'] = authority;
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            return data;
        },
        ration_card(text) {
            const data = {};
            const ration = text.match(PATTERNS.RATION_CARD);
            if (ration) data['Ration Card Number'] = ration[1].trim();
            const name = nameOrFallback(text);
            if (name) data['Household Head Name'] = name;
            const members = extractField(text, /(?:members|family)\s*(?:members)?\s*[:]?\s*(\d+)/i);
            if (members) data['Family Member Count'] = members;
            const category = extractField(text, /(?:category|type|color)\s*[:]?\s*([A-Za-z ]+)/i);
            if (category) data['Category'] = category;
            const addr = text.match(PATTERNS.ADDRESS);
            if (addr) data['Address'] = addr[1].trim();
            return data;
        },
        school_leaving_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Student Name'] = name;
            const dob = text.match(PATTERNS.DATE);
            if (dob) data['Date of Birth'] = dob[0];
            const place = extractField(text, /place of birth\s*[:]?\s*([A-Za-z ]+)/i);
            if (place) data['Place of Birth'] = place;
            const religion = extractField(text, /religion\s*[:]?\s*([A-Za-z ]+)/i);
            if (religion) data['Religion'] = religion;
            const caste = text.match(PATTERNS.CASTE);
            if (caste) data['Caste'] = caste[1].trim();
            const leavingDate = text.match(PATTERNS.DATE);
            if (leavingDate) data['Date of Leaving'] = leavingDate[0];
            const school = extractField(text, /(?:school|institution)\s*[:]?\s*([A-Za-z ]+)/i);
            if (school) data['School Name'] = school;
            return data;
        },
        ssc_marksheet(text) { return marksheetLike(text, 'Student Name'); },
        hsc_marksheet(text) { return marksheetLike(text, 'Student Name'); },
        degree_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Student Name'] = name;
            const serial = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (serial) data['Degree Serial Number'] = serial[1].trim();
            const year = text.match(PATTERNS.YEAR);
            if (year) data['Passing Year'] = year[0];
            const course = extractField(text, /(?:course|degree|program)\s*[:]?\s*([A-Za-z ]+)/i);
            if (course) data['Course Name'] = course;
            const classification = extractField(text, /(?:classification|class|grade)\s*[:]?\s*([A-Za-z0-9 ]+)/i);
            if (classification) data['Classification/Class'] = classification;
            const university = extractField(text, /(?:university|institution)\s*[:]?\s*([A-Za-z ]+)/i);
            if (university) data['University Name'] = university;
            return data;
        },
        board_passing_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Student Name'] = name;
            const year = text.match(PATTERNS.YEAR);
            if (year) data['Passing Year'] = year[0];
            const board = extractField(text, /(?:board|university)\s*[:]?\s*([A-Za-z ]+)/i);
            if (board) data['Board/University Name'] = board;
            const total = extractField(text, /(?:total|obtained)\s*(?:marks)?\s*[:]?\s*([0-9/ ]+)/i);
            if (total) data['Total Marks Obtained'] = total;
            const result = extractField(text, /(?:result|status)\s*[:]?\s*([A-Za-z ]+)/i);
            if (result) data['Result Status'] = result;
            return data;
        },
        caste_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const caste = text.match(PATTERNS.CASTE);
            if (caste) data['Caste Category'] = caste[1].trim();
            const subCaste = extractField(text, /sub[-\s]caste\s*[:]?\s*([A-Za-z ]+)/i);
            if (subCaste) data['Sub-Caste Name'] = subCaste;
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Outward Number'] = cert[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            return data;
        },
        caste_validity_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const decision = extractField(text, /(?:decision|verdict)\s*[:]?\s*([A-Za-z ]+)/i);
            if (decision) data['Scrutiny Committee Decision'] = decision;
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Validity Certificate Number'] = cert[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            const caseNo = extractField(text, /case\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)/i);
            if (caseNo) data['Case Number'] = caseNo;
            return data;
        },
        non_creamy_layer_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const year = text.match(PATTERNS.YEAR);
            if (year) data['Financial Year'] = year[0];
            const subCaste = extractField(text, /sub[-\s]caste\s*[:]?\s*([A-Za-z ]+)/i);
            if (subCaste) data['OBC Sub-Caste'] = subCaste;
            const validity = text.match(PATTERNS.VALIDITY);
            if (validity) data['Validity Expiry Date'] = validity[1].trim();
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Certificate Number'] = cert[1].trim();
            return data;
        },
        ews_certificate(text) {
            const data = {};
            const name = nameOrFallback(text);
            if (name) data['Applicant Name'] = name;
            const assets = extractField(text, /(?:asset|property|valuation)\s*[:]?\s*([A-Za-z0-9\-/ ]+)/i);
            if (assets) data['Asset Valuation Details'] = assets;
            const income = text.match(PATTERNS.INCOME);
            if (income) data['Annual Income Limit Verification'] = income[1].trim();
            const cert = text.match(PATTERNS.CERTIFICATE_NUMBER);
            if (cert) data['Certificate Number'] = cert[1].trim();
            const issue = text.match(PATTERNS.ISSUE_DATE);
            if (issue) data['Date of Issue'] = issue[1].trim();
            return data;
        },
        gst_certificate(text) {
            const data = {};
            const gst = text.match(PATTERNS.GST);
            if (gst) data['GST Number'] = gst[0];
            const name = nameOrFallback(text);
            if (name) data['Business Name'] = name;
            const businessType = extractField(text, /(?:business|type)\s*[:]?\s*([A-Za-z ]+)/i);
            if (businessType) data['Business Type'] = businessType;
            const regDate = text.match(PATTERNS.DATE);
            if (regDate) data['Date of Registration'] = regDate[0];
            const state = text.match(PATTERNS.DISTRICT);
            if (state) data['State'] = state[1].trim();
            const addr = text.match(PATTERNS.ADDRESS);
            if (addr) data['Address'] = addr[1].trim();
            const status = extractField(text, /status\s*[:]?\s*([A-Za-z ]+)/i);
            if (status) data['Status'] = status;
            return data;
        },
        invoice(text) {
            const data = {};
            const invoice = text.match(PATTERNS.INVOICE);
            if (invoice) data['Invoice Number'] = invoice[1].trim();
            const invDate = text.match(PATTERNS.DATE);
            if (invDate) data['Invoice Date'] = invDate[0];
            const customer = text.match(PATTERNS.NAME);
            if (customer) data['Customer Name'] = customer[1].trim();
            const customerGst = text.match(PATTERNS.GST);
            if (customerGst) data['Customer GST'] = customerGst[0];
            const total = text.match(PATTERNS.TOTAL);
            if (total) data['Total Amount'] = total[1].trim();
            const tax = extractField(text, /(?:tax|gst|vat)\s*(?:amount)?\s*[:]?\s*[\u20B9$]?\s*([\d,]+\.?\d*)/i);
            if (tax) data['Tax Amount'] = tax;
            return data;
        },
    };

    function marksheetLike(text, nameLabel) {
        const data = {};
        const name = nameOrFallback(text);
        if (name) data[nameLabel] = name;
        const seat = text.match(PATTERNS.SEAT_NUMBER);
        if (seat) data['Seat Number'] = seat[1].trim();
        const percentage = text.match(PATTERNS.PERCENTAGE);
        if (percentage) data['Total Percentage'] = percentage[1] + '%';
        const year = text.match(PATTERNS.YEAR);
        if (year) data['Year'] = year[0];
        const division = extractField(text, /(?:division|grade|class)\s*[:]?\s*([A-Za-z0-9 ]+)/i);
        if (division) data['Division/Grade'] = division;
        return data;
    }

    function genericExtraction(text) {
        if (!text || typeof text !== 'string') return {};
        const data = {};
        const email = text.match(PATTERNS.EMAIL);
        if (email) data['Email'] = email[0];
        const phone = text.match(PATTERNS.PHONE);
        if (phone) data['Phone'] = phone[0];
        const name = nameOrFallback(text);
        if (name) data['Name'] = name;
        const date = text.match(PATTERNS.DATE);
        if (date) data['Date'] = date[0];
        const ref = text.match(PATTERNS.CERTIFICATE_NUMBER);
        if (ref) data['Reference Number'] = ref[1].trim();
        return data;
    }

    // ============================================================
    //  MAIN ENTRY POINT (mirrors extract_structured_data)
    // ============================================================
    function extractStructuredData(text) {
        if (!text || typeof text !== 'string') return {};
        const cleaned = cleanOcrText(text);
        if (cleaned.length < MIN_TEXT_LENGTH_FOR_EXTRACTION) return {};

        const docType = detectDocumentType(cleaned);
        if (!docType) return genericExtraction(cleaned);

        const extractor = EXTRACTORS[docType];
        if (extractor) return extractor(cleaned);
        return genericExtraction(cleaned);
    }

    function mergePageData(pages) {
        const merged = {};
        pages.forEach((page) => {
            Object.keys(page).forEach((key) => {
                if (key === '_metadata') return;
                if (!(key in merged)) {
                    merged[key] = page[key];
                } else if (Array.isArray(merged[key]) && Array.isArray(page[key])) {
                    merged[key] = merged[key].concat(page[key]);
                } else if (typeof merged[key] === 'string' && typeof page[key] === 'string') {
                    merged[key] = `${merged[key]} ${page[key]}`;
                }
            });
        });
        return merged;
    }

    global.offlineExtract = {
        cleanOcrText,
        maskSensitiveData,
        detectDocumentType,
        extractStructuredData,
        genericExtraction,
        mergePageData,
        PATTERNS,
    };
})(window);
