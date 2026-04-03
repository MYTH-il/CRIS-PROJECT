from fpdf import FPDF
from datetime import datetime

class IntelligenceDossier(FPDF):
    def header(self):
        # Official agency red branding
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(180, 0, 0)
        self.cell(0, 10, 'CRIS OFFICIAL INTELLIGENCE DOSSIER', 0, 1, 'C')
        
        # Timestamp
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(220, 230, 240)
        self.set_text_color(0, 0, 50)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        # encode/decode to handle any weird ascii characters
        clean_body = str(body).encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 6, clean_body)
        self.ln(6)

def create_pdf_dossier(incident_text, prediction, confidence, severity, entities, matches, future_threat):
    """
    Builds the PDF bytecode string directly in python memory.
    """
    pdf = IntelligenceDossier()
    pdf.add_page()
    
    # 1. Incident Narrative
    pdf.chapter_title('1. RAW INCIDENT NARRATIVE')
    pdf.chapter_body(incident_text)
    
    # 2. Base Classification & Triage
    pdf.chapter_title('2. NEURAL NETWORK CLASSIFICATION & TRIAGE')
    triage_txt = f"Detected Crime Type: {prediction}\n"
    triage_txt += f"Neural Network Confidence: {confidence:.2f}%\n"
    triage_txt += f"VADER Severity Triage Level: {severity}/10\n"
    
    if severity >= 8:
        triage_txt += ">>> STRATEGIC ALERT: CODE RED (Extreme Distress/Weapons/High-Priority)"
    elif severity >= 5:
        triage_txt += ">>> STRATEGIC ALERT: ELEVATED THREAT"
    else:
        triage_txt += ">>> STRATEGIC ALERT: STANDARD PRIORITY"
        
    pdf.chapter_body(triage_txt)
    
    # 3. Entities
    pdf.chapter_title('3. EXTRACTED INTELLIGENCE ENTITIES')
    if entities and len(entities) > 0:
        ent_text = ""
        for category, items in entities.items():
            ent_text += f"{category}:\n"
            for item in items:
                ent_text += f" - {item}\n"
        pdf.chapter_body(ent_text)
    else:
        pdf.chapter_body("No actionable entities (Persons, Organizations, Locations) detected.")

    # 4. PREDICTIVE FORECASTING
    pdf.chapter_title('4. RANDOM FOREST ML PREDICTIVE FORECAST')
    if future_threat:
        pdf.chapter_body(future_threat)
    else:
        pdf.chapter_body("Warning: Predictive ML Engine offline. No future risk assessment generated.")

    # 5. MO Pattern matching
    pdf.chapter_title('5. MODUS OPERANDI (M.O.) HISTORICAL MATCHES')
    if matches and len(matches) > 0:
        mo_txt = ""
        for i, m in enumerate(matches, 1):
            mo_txt += f"--- MATCH {i} [Similarity: {m['similarity']:.1f}% | Crime: {m['crime']}] ---\n"
            mo_txt += f"{m['snippet']}\n\n"
        pdf.chapter_body(mo_txt)
    else:
        pdf.chapter_body("No M.O. historical database matches found.")
        
    # Return binary PDF data buffer as standard bytes instead of bytearray
    return bytes(pdf.output())
