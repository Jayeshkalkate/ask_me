from django.db import models
from django.contrib.auth.models import User
import numpy as np


def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types for JSONField."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):  # handles np.bool_, np.int64, etc.
        return obj.item()
    return obj


class Document(models.Model):
    """
    Model to store uploaded documents with OCR data and dynamic extracted fields.
    """

    DOC_TYPES = [
        ("aadhaar_card", "Aadhaar Card"),
        ("pan_card", "PAN Card"),
        ("driving_license", "Driving License"),
        ("voter_id_card", "Voter ID Card"),
        ("vehicle_registration_certificate", "Vehicle Registration Certificate"),
        ("domicile_certificate", "Domicile Certificate"),
        ("bank_passbook", "Bank Passbook"),
        ("birth_certificate", "Birth Certificate"),
        ("income_certificate", "Income Certificate"),
        ("caste_certificate", "Caste Certificate"),
        ("cast_validity_certificate", "Cast Validity Certificate"),
        ("non_criminal_certificate", "Non-Criminal Certificate"),
        (
            "marksheets_passing_certificates",
            "10th/12th Marksheet & Passing Certificate",
        ),
        ("diploma_degree_certificate", "Diploma / Degree / Provisional Certificate"),
        ("migration_certificate", "Migration Certificate"),
        ("leaving_transfer_certificate", "Leaving / Transfer Certificate"),
        ("bonafide_certificate", "Bonafide Certificate"),
        ("passport", "Passport"),
        ("ration_card", "Ration Card"),
        ("aadhaar_enrollment_receipt", "Aadhaar Enrollment Receipt"),
        ("bank_kyc_documents", "Bank KYC Documents"),
        ("electricity_bill", "Electricity Bill"),
        ("telephone_bill", "Telephone Bill"),
        ("property_sale_deed", "Property Sale Deed"),
        ("rent_agreement", "Rent Agreement"),
        ("insurance_policy", "Insurance Policy"),
        ("medical_records", "Medical Records"),
        ("vehicle_insurance_certificate", "Vehicle Insurance Certificate"),
        ("form16_salary_slip_it_returns", "Form 16 / Salary Slip / IT Returns"),
        ("ews_minority_certificate", "EWS / Minority Certificate"),
        (
            "trade_license_business_registration",
            "Trade License / Business Registration",
        ),
        ("employment_certificate", "Employment Certificate"),
        (
            "police_clearance_character_certificate",
            "Police Clearance / Character Certificate",
        ),
        ("driving_test_certificate", "Driving Test Certificate"),
        ("student_college_id", "Student / College ID"),
        ("income_tax_acknowledgment_receipt", "Income Tax Acknowledgment Receipt"),
        ("pension_social_security_documents", "Pension / Social Security Documents"),
        ("loan_mortgage_documents", "Loan / Mortgage Documents"),
        ("passport_application_receipt", "Passport Application Receipt"),
        ("visa_oci_pio_documents", "Visa / OCI / PIO Documents"),
        (
            "birth_death_marriage_divorce_certificate",
            "Birth / Death / Marriage / Divorce Certificate",
        ),
        ("adoption_guardianship_certificate", "Adoption / Guardianship Certificate"),
        ("affidavit", "Affidavit"),
        ("professional_government_license", "Professional / Government License"),
        ("election_commission_acknowledgment", "Election Commission Acknowledgment"),
        ("other_document", "Other Document"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    file = models.FileField(upload_to="documents/")
    doc_type = models.CharField(
        max_length=50,
        choices=DOC_TYPES,
        default="other_document",
        blank=True,
        null=True,
    )

    # Raw OCR text
    extracted_text = models.TextField(blank=True, null=True)

    # Structured extracted data (from OCR)
    extracted_data = models.JSONField(default=dict, blank=True)

    # User-edited structured data
    user_edited_data = models.JSONField(default=dict, blank=True)

    processed = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, null=True)
    quality_score = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        if self.doc_type:
            return f"{self.get_doc_type_display()} - {self.user.username} ({self.created_at.strftime('%Y-%m-%d')})"
        return f"{self.user.username} - {self.file.name}"


# ✅ Document field templates for automatic extracted_data initialization
DOCUMENT_FIELD_TEMPLATES = {
    "aadhaar_card": {
        "Full Name": "",
        "Aadhaar Number": "",
        "Date of Birth / Age": "",
        "Gender": "",
        "Address": "",
    },
    "pan_card": {
        "Full Name": "",
        "PAN Number": "",
        "Father's Name": "",
        "Date of Birth": "",
    },
    "driving_license": {
        "License Number": "",
        "Name of Holder": "",
        "Date of Birth": "",
        "Date of Issue / Expiry": "",
        "Address": "",
        "Vehicle Class Allowed": "",
    },
    "voter_id_card": {
        "Name of Holder": "",
        "Father / Husband Name": "",
        "Voter ID Number": "",
        "Date of Birth / Age": "",
        "Gender": "",
        "Address": "",
        "Electoral Roll / Assembly Constituency": "",
    },
    "vehicle_registration_certificate": {
        "Registration Number": "",
        "Vehicle Make & Model": "",
        "Engine Number & Chassis Number": "",
        "Owner Name & Address": "",
        "Registration Date": "",
        "Fitness Certificate": "",
        "Vehicle Type & Class": "",
    },
    "domicile_certificate": {
        "Name of Holder": "",
        "Father / Husband Name": "",
        "Date of Birth / Age": "",
        "Permanent Address": "",
        "State / District of Residence": "",
        "Certificate Issue Date": "",
    },
    "bank_passbook": {
        "Account Holder Name": "",
        "Account Number": "",
        "Bank Name & Branch": "",
        "IFSC Code": "",
        "Transaction History": "",
    },
    "birth_certificate": {
        "Name of Child": "",
        "Date & Place of Birth": "",
        "Gender": "",
        "Parents' Names": "",
        "Registration Number": "",
        "Birth Weight / Hospital Name": "",
    },
    "income_certificate": {
        "Name of Applicant": "",
        "Father / Husband Name": "",
        "Date of Birth / Age": "",
        "Address": "",
        "Annual Income": "",
        "Certificate Issue Date": "",
    },
    "caste_certificate": {
        "Name of Applicant": "",
        "Father / Husband Name": "",
        "Date of Birth / Age": "",
        "Address": "",
        "Caste Name": "",
        "Sub-caste": "",
        "Certificate Issue Date": "",
    },
    "non_criminal_certificate": {
        "Name of Applicant": "",
        "Father / Husband Name": "",
        "Date of Birth / Age": "",
        "Address": "",
        "Certificate Validity": "",
        "Purpose of Certificate": "",
        "Issuing Authority & Date": "",
    },
    "marksheets_passing_certificates": {
        "Student Name": "",
        "Roll Number / Registration Number": "",
        "School / Board Name": "",
        "Subject-wise Marks": "",
        "Total Marks & Percentage / CGPA": "",
        "Grade": "",
        "Date of Issue": "",
    },
    "diploma_degree_certificate": {
        "Student Name": "",
        "Roll / Registration Number": "",
        "College / University Name": "",
        "Course Name": "",
        "Semester-wise Marks & Grades": "",
        "Total Marks / CGPA": "",
        "Year of Passing": "",
        "Date of Issue": "",
    },
    "migration_certificate": {
        "Student Name": "",
        "Father / Mother Name": "",
        "Previous School / University": "",
        "Course Name": "",
        "Year of Completion": "",
        "Reason for Migration": "",
        "Date of Issue": "",
    },
    "leaving_transfer_certificate": {
        "Student Name": "",
        "Father / Mother Name": "",
        "Date of Birth / Admission": "",
        "Class / Course Last Attended": "",
        "Date of Leaving": "",
        "Reason for Leaving / Transfer": "",
        "School / College Name": "",
    },
    "bonafide_certificate": {
        "Student / Resident Name": "",
        "Father / Guardian Name": "",
        "Date of Birth / Admission": "",
        "Purpose of Certificate": "",
        "Institution Name": "",
        "Period of Residence / Enrollment": "",
        "Date of Issue": "",
    },
    "passport": {
        "Name of Passport Holder": "",
        "Passport Number": "",
        "Nationality": "",
        "Date of Birth / Place of Birth": "",
        "Gender": "",
        "Date of Issue / Expiry": "",
        "Place of Issue": "",
    },
    "ration_card": {
        "Head of Family Name": "",
        "Members Names": "",
        "Card Number": "",
        "Address": "",
        "Category": "",
    },
    "aadhaar_enrollment_receipt": {
        "Applicant Name": "",
        "Enrollment ID": "",
        "Date of Enrollment": "",
        "Center Name": "",
    },
    "bank_kyc_documents": {
        "Account Holder Name": "",
        "Account Number": "",
        "Bank Name & Branch": "",
        "IFSC Code": "",
        "KYC Documents Submitted": "",
    },
    "electricity_bill": {
        "Consumer Name": "",
        "Bill Number": "",
        "Address": "",
        "Bill Date": "",
        "Due Date": "",
        "Amount": "",
    },
    "telephone_bill": {
        "Subscriber Name": "",
        "Account / Connection Number": "",
        "Address": "",
        "Bill Date": "",
        "Due Date": "",
        "Amount": "",
    },
    "property_sale_deed": {
        "Owner Name": "",
        "Property Address": "",
        "Survey / Plot Number": "",
        "Sale Date": "",
        "Sale Value": "",
        "Witness Names": "",
    },
    "rent_agreement": {
        "Tenant Name": "",
        "Landlord Name": "",
        "Property Address": "",
        "Lease Start / End Date": "",
        "Rent Amount": "",
        "Security Deposit": "",
    },
    "insurance_policy": {
        "Policy Holder Name": "",
        "Policy Number": "",
        "Type of Policy": "",
        "Issue Date": "",
        "Expiry Date": "",
        "Coverage Amount": "",
    },
    "medical_records": {
        "Patient Name": "",
        "Date of Birth": "",
        "Hospital / Clinic Name": "",
        "Doctor Name": "",
        "Diagnosis": "",
        "Prescription / Treatment": "",
    },
    "vehicle_insurance_certificate": {
        "Policy Holder Name": "",
        "Vehicle Registration Number": "",
        "Policy Number": "",
        "Issue Date": "",
        "Expiry Date": "",
        "Insurance Company": "",
    },
    "form16_salary_slip_it_returns": {
        "Employee Name": "",
        "PAN Number": "",
        "Employer Name": "",
        "Financial Year": "",
        "Total Income": "",
        "Tax Deducted / Paid": "",
    },
    "ews_minority_certificate": {
        "Applicant Name": "",
        "Father / Guardian Name": "",
        "Date of Birth / Age": "",
        "Address": "",
        "Certificate Issue Date": "",
        "Purpose": "",
    },
    "trade_license_business_registration": {
        "Business Name": "",
        "Owner Name": "",
        "Registration Number": "",
        "Issue Date": "",
        "Expiry Date": "",
        "Issuing Authority": "",
    },
    "employment_certificate": {
        "Employee Name": "",
        "Father / Guardian Name": "",
        "Date of Birth": "",
        "Organization Name": "",
        "Designation / Role": "",
        "Period of Employment": "",
    },
    "police_clearance_character_certificate": {
        "Applicant Name": "",
        "Father / Guardian Name": "",
        "Date of Birth": "",
        "Address": "",
        "Purpose": "",
        "Certificate Validity": "",
        "Issuing Authority": "",
    },
    "driving_test_certificate": {
        "Candidate Name": "",
        "License Number": "",
        "Test Date": "",
        "Test Center": "",
        "Result": "",
    },
    "student_college_id": {
        "Student Name": "",
        "College / School Name": "",
        "ID Number": "",
        "Course / Class": "",
        "Validity Date": "",
    },
    "income_tax_acknowledgment_receipt": {
        "PAN Number": "",
        "Acknowledgment Number": "",
        "Financial Year": "",
        "Date of Filing": "",
        "Tax Paid": "",
    },
    "pension_social_security_documents": {
        "Pensioner Name": "",
        "Father / Guardian Name": "",
        "Date of Birth": "",
        "Pension / Social Security Number": "",
        "Issuing Authority": "",
        "Period / Validity": "",
    },
    "loan_mortgage_documents": {
        "Applicant Name": "",
        "Father / Guardian Name": "",
        "Loan Account Number": "",
        "Bank / Financial Institution": "",
        "Loan Amount": "",
        "Issue Date": "",
        "Tenure": "",
    },
    "passport_application_receipt": {
        "Applicant Name": "",
        "Application Number": "",
        "Date of Application": "",
        "Passport Office": "",
    },
    "visa_oci_pio_documents": {
        "Applicant Name": "",
        "Document Number": "",
        "Nationality": "",
        "Date of Issue / Expiry": "",
        "Purpose / Type": "",
    },
    "birth_death_marriage_divorce_certificate": {
        "Person Name": "",
        "Date of Event": "",
        "Place of Event": "",
        "Parents / Spouse Names": "",
        "Registration Number": "",
        "Issuing Authority": "",
    },
    "adoption_guardianship_certificate": {
        "Child Name": "",
        "Guardian / Parent Name": "",
        "Date of Birth": "",
        "Adoption / Guardianship Date": "",
        "Issuing Authority": "",
    },
    "affidavit": {
        "Applicant Name": "",
        "Purpose / Subject": "",
        "Date of Issue": "",
        "Issuing Authority": "",
    },
    "professional_government_license": {
        "License Holder Name": "",
        "License Number": "",
        "Issuing Authority": "",
        "Issue Date": "",
        "Expiry Date": "",
        "Profession / Field": "",
    },
    "election_commission_acknowledgment": {
        "Applicant Name": "",
        "Father / Guardian Name": "",
        "Date of Birth": "",
        "Voter ID / Application Number": "",
        "Date of Registration": "",
        "Issuing Authority": "",
    },
    "other_document": {"Additional Fields": ""},
}
