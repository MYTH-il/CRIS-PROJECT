"""
Crime Report Intelligence System — Synthetic Dataset Generator
Generates ~60,000 realistic India-context crime report records.
Run: python generate_crime_dataset.py
Output: crime_reports_synthetic.csv
"""

import pandas as pd
import numpy as np
import random
import uuid
import json
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

NUM_RECORDS = 60000

# ─── Reference Data ────────────────────────────────────────────────────────────

STATES_DISTRICTS = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Thane", "Ratnagiri", "Solapur"],
    "Delhi": ["Central Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi", "New Delhi"],
    "Uttar Pradesh": ["Lucknow", "Agra", "Kanpur", "Varanasi", "Allahabad", "Meerut", "Noida"],
    "Karnataka": ["Bengaluru Urban", "Bengaluru Rural", "Mysuru", "Hubli", "Mangaluru", "Belagavi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Tirunelveli"],
    "West Bengal": ["Kolkata", "Howrah", "Asansol", "Siliguri", "Durgapur", "Bardhaman"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer", "Bikaner"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Gandhinagar"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga", "Purnia"],
}

STATION_CODES = [f"PS{str(i).zfill(4)}" for i in range(1, 501)]
OFFICER_IDS = [f"OFF{str(i).zfill(5)}" for i in range(1, 2001)]

CRIME_TYPES = {
    "Theft": {
        "subtypes": ["Burglary", "Pickpocketing", "Vehicle Theft", "Chain Snatching", "Shoplifting", "House Breaking"],
        "weight": 0.22,
        "severity": ["Low", "Medium"],
        "weapons": ["None", "Knife"],
        "descriptions": [
            "The complainant reported that unknown persons broke into the premises during night hours and stole valuables including gold ornaments and cash.",
            "Victim states that while travelling on a crowded bus, an unknown person snatched the gold chain from their neck and fled.",
            "Owner of the vehicle parked near the market found the vehicle missing in the morning. CCTV footage being reviewed.",
            "Shop owner reported that during closing hours, two persons entered the shop and took merchandise worth substantial amount without payment.",
            "Complainant noticed the absence of wallet containing cash and identity documents after returning from the railway station.",
        ]
    },
    "Assault": {
        "subtypes": ["Simple Assault", "Grievous Hurt", "Domestic Violence", "Road Rage", "Mob Violence"],
        "weight": 0.18,
        "severity": ["Medium", "High"],
        "weapons": ["None", "Knife", "Rod", "Blunt Object"],
        "descriptions": [
            "The complainant was attacked by a group of individuals following an argument over a property dispute. Sustained injuries on face and arms.",
            "Victim reported being assaulted by the accused with a blunt instrument near the market area. Multiple witnesses present.",
            "Domestic dispute escalated and the accused physically assaulted the victim inside the residence. Neighbours called the police.",
            "Road rage incident near the junction resulted in an altercation where the accused broke the windshield and assaulted the driver.",
            "Victim was returning home late at night when accosted by three individuals who demanded money and beat them upon refusal.",
        ]
    },
    "Fraud": {
        "subtypes": ["Cyber Fraud", "Bank Fraud", "Identity Theft", "Investment Scam", "Cheating", "Property Fraud"],
        "weight": 0.16,
        "severity": ["Medium", "High"],
        "weapons": ["None"],
        "descriptions": [
            "Complainant received a call from an individual claiming to be a bank executive. Victim shared OTP details and lost a significant amount from bank account.",
            "Victim invested money in a scheme promising high returns. The accused collected funds from multiple investors and absconded.",
            "Forged documents were used to fraudulently transfer property ownership without the knowledge or consent of the rightful owner.",
            "Online shopping fraud where victim paid for goods that were never delivered. The seller profile on the platform has since been deactivated.",
            "Accused impersonated a government official and collected money from the complainant under the pretext of providing a government job.",
        ]
    },
    "Murder": {
        "subtypes": ["Premeditated Murder", "Culpable Homicide", "Dowry Death", "Contract Killing", "Honour Killing"],
        "weight": 0.05,
        "severity": ["Critical"],
        "weapons": ["Knife", "Firearm", "Poison", "Blunt Object", "Strangulation"],
        "descriptions": [
            "Body of the deceased was discovered with multiple stab wounds in the residential area. The victim was identified by a family member.",
            "Following a prolonged dispute over ancestral property, the accused allegedly planned and executed the murder of the complainant's kin.",
            "Deceased was found in the field with firearm injuries. Preliminary investigation suggests involvement of rival faction.",
            "Victim was last seen alive two days prior to the discovery of the body. Cause of death determined as blunt force trauma to the head.",
            "Accused allegedly administered poison to the victim following domestic disputes. Post mortem report pending confirmation.",
        ]
    },
    "Kidnapping": {
        "subtypes": ["Child Abduction", "Ransom Kidnapping", "Trafficking", "Abduction for Marriage", "Custodial Abduction"],
        "weight": 0.07,
        "severity": ["High", "Critical"],
        "weapons": ["None", "Firearm", "Knife"],
        "descriptions": [
            "Parents reported the disappearance of the minor child who went missing on the way back from school. Last seen near the bus stop.",
            "The kidnapped individual was held captive and the accused demanded a substantial ransom from the family via phone calls.",
            "Woman was lured under the pretext of a job offer and was subsequently found to have been transported to another state.",
            "Complainant alleges that the accused forcibly took away their daughter against her will and performed an illegal marriage.",
            "Child went missing from the playground. Witnesses reported seeing a stranger talking to the child before the disappearance.",
        ]
    },
    "Robbery": {
        "subtypes": ["Armed Robbery", "Bank Robbery", "Dacoity", "Highway Robbery", "ATM Robbery"],
        "weight": 0.09,
        "severity": ["High", "Critical"],
        "weapons": ["Firearm", "Knife", "Rod"],
        "descriptions": [
            "Armed individuals entered the establishment and held staff at gunpoint. Cash from the till and valuables were taken before fleeing.",
            "Victim was returning from the bank when two individuals on a motorcycle snatched the bag containing cash at knifepoint.",
            "A group of armed persons waylaid the vehicle on the highway and robbed passengers of cash and valuables at gunpoint.",
            "ATM kiosk was targeted in the early morning hours. CCTV footage shows three masked individuals breaking into the machine.",
            "The accused entered the premises with weapons and tied up the occupants before looting the house systematically.",
        ]
    },
    "Drug Offence": {
        "subtypes": ["Possession", "Trafficking", "Manufacturing", "Peddling", "Consumption"],
        "weight": 0.08,
        "severity": ["Medium", "High"],
        "weapons": ["None"],
        "descriptions": [
            "During routine patrolling, officers intercepted the accused who was found in possession of contraband substance concealed in a bag.",
            "Based on a tip-off, a raid was conducted at the premises where a significant quantity of narcotics was discovered and seized.",
            "Accused was found peddling drugs near the school premises. Caught red-handed during an undercover operation by the narcotic cell.",
            "Seizure of heroin from the vehicle during a check post operation. The accused was transporting the contraband across district lines.",
            "Search of the warehouse revealed a makeshift laboratory used for manufacturing synthetic drugs. Three accused arrested.",
        ]
    },
    "Cybercrime": {
        "subtypes": ["Hacking", "Online Harassment", "Child Pornography", "Phishing", "Ransomware", "Social Media Fraud"],
        "weight": 0.07,
        "severity": ["Medium", "High"],
        "weapons": ["None"],
        "descriptions": [
            "Complainant reports receiving threatening and obscene messages from an unknown account on social media. Screenshots submitted as evidence.",
            "Company server was hacked and sensitive data of thousands of customers was compromised. Ransom demand was made by the attacker.",
            "Victim received a phishing email that appeared to be from a legitimate bank. Account credentials were stolen leading to financial loss.",
            "Accused created a fake profile in the name of the complainant and posted defamatory content damaging reputation.",
            "Business email compromise resulted in fraudulent transfer of funds. The email account of the senior executive was impersonated.",
        ]
    },
    "Sexual Offence": {
        "subtypes": ["Rape", "Molestation", "Stalking", "Eve Teasing", "Sexual Harassment at Workplace"],
        "weight": 0.05,
        "severity": ["High", "Critical"],
        "weapons": ["None", "Knife"],
        "descriptions": [
            "Victim reported being followed persistently by the accused despite clear refusal. Accused appeared at the victim's workplace multiple times.",
            "Incident occurred at the workplace. Victim alleges the accused made repeated inappropriate advances and created a hostile environment.",
            "Complainant reports that an unknown individual touched inappropriately in a public space. Occurred near the market area.",
            "Victim filed a complaint after being accosted near the isolated stretch of road during late evening hours.",
            "Accused sent unsolicited explicit messages and images repeatedly to the complainant's phone number. Call records as evidence.",
        ]
    },
    "Property Damage": {
        "subtypes": ["Vandalism", "Arson", "Mischief", "Trespass", "Encroachment"],
        "weight": 0.03,
        "severity": ["Low", "Medium"],
        "weapons": ["None", "Blunt Object"],
        "descriptions": [
            "Unknown persons damaged the complainant's vehicle parked outside the residential premises during the night.",
            "Fire was set to the farmland and crop was destroyed. Complainant suspects involvement of a rival party due to ongoing land dispute.",
            "Accused trespassed into the property and caused damage to the boundary wall and gates. Neighbours witnessed the incident.",
            "Shop shutters were vandalized and the signboard was damaged. The complainant suspects business rivalry as the motive.",
            "Encroachment on the land belonging to the complainant. The accused constructed a structure on disputed land without legal sanction.",
        ]
    },
}

AREA_TYPES = ["Urban", "Semi-Urban", "Rural", "Industrial", "Commercial", "Residential", "Highway"]
FILING_METHODS = ["In-Person", "Online Portal", "Phone", "WhatsApp Helpline", "Walk-in"]
LANDMARKS = [
    "Near Railway Station", "Adjacent to Market", "Behind Petrol Pump", "Near School",
    "Opposite Hospital", "Near Highway", "Inside Residential Colony", "Near Temple/Mosque",
    "Near ATM", "Near Bus Stand", "Adjacent to Park", "Near Bank Branch"
]
OCCUPATIONS = [
    "Farmer", "Business Owner", "Government Employee", "Private Sector Employee",
    "Student", "Homemaker", "Daily Wage Labourer", "Self-Employed", "Retired",
    "Driver", "Teacher", "Shop Owner", "Unemployed", "Healthcare Worker"
]
EVIDENCE_TYPES = ["CCTV Footage", "Witness Testimony", "Forensic Report", "Phone Records",
                  "Financial Records", "Physical Evidence", "Digital Evidence", "None"]
MODUS_OPERANDI = [
    "Broke lock", "Distraction technique", "Impersonation", "Force", "Deception",
    "Social engineering", "Hacking", "Drug-induced", "Targeted attack", "Opportunistic"
]
TIME_OF_DAY = ["Early Morning (4-7am)", "Morning (7-11am)", "Afternoon (11am-3pm)",
               "Evening (3-7pm)", "Night (7-11pm)", "Late Night (11pm-4am)"]
WITNESS_TEMPLATES = [
    "Witness states that they observed the accused near the scene approximately {time} before the incident was reported.",
    "Deponent confirms seeing a {gender} individual matching the description of the accused leaving the area in haste.",
    "Witness was present at the location and confirms the sequence of events as described by the complainant.",
    "Deponent states they heard loud noises from the adjacent premises and called emergency services.",
    "Witness says the accused was known to them from the neighbourhood and was seen at the location on the day of the incident.",
    "No eyewitness to the incident. Complaint based solely on discovery of the offence.",
    "Deponent is a passer-by who noticed suspicious activity and alerted the nearby police patrol.",
]
OFFICER_REMARKS_TEMPLATES = [
    "Complaint registered. Scene of crime visited and panchanama prepared. Investigating Officer assigned.",
    "Case appears prima facie genuine. Further investigation underway. Accused absconding.",
    "One accused apprehended. Remaining accused being traced. Charge sheet to be filed.",
    "CCTV footage recovered and under analysis. Technical team called in.",
    "Victim referred to medical examination. Forensic team dispatched to the scene.",
    "Case transferred to Crime Branch for specialized investigation.",
    "Complaint registered under relevant IPC sections. Preliminary investigation completed.",
    "Matter under investigation. Complainant advised to produce additional documents.",
]
ACTION_TAKEN = [
    "FIR Registered, Accused Arrested", "FIR Registered, Investigation Ongoing",
    "FIR Registered, Accused Absconding", "Closure Report Filed",
    "Case Transferred to Crime Branch", "Charge Sheet Filed", "Under Trial"
]

# ─── Generation Helpers ────────────────────────────────────────────────────────

def random_date(start_year=2018, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def generate_lat_lon(state):
    bounds = {
        "Maharashtra": (15.6, 22.0, 72.6, 80.9),
        "Delhi": (28.4, 28.9, 76.8, 77.4),
        "Uttar Pradesh": (23.8, 30.4, 77.1, 84.6),
        "Karnataka": (11.5, 18.4, 74.0, 78.6),
        "Tamil Nadu": (8.0, 13.6, 76.2, 80.4),
        "West Bengal": (21.5, 27.2, 85.8, 89.9),
        "Rajasthan": (23.0, 30.2, 69.3, 78.3),
        "Gujarat": (20.1, 24.7, 68.2, 74.5),
        "Madhya Pradesh": (21.0, 26.9, 74.0, 82.8),
        "Bihar": (24.3, 27.5, 83.3, 88.3),
    }
    lat_min, lat_max, lon_min, lon_max = bounds.get(state, (20.0, 28.0, 72.0, 85.0))
    return round(random.uniform(lat_min, lat_max), 6), round(random.uniform(lon_min, lon_max), 6)

def generate_incident_description(crime_type, subtype, area_type, district, weapon):
    base = random.choice(CRIME_TYPES[crime_type]["descriptions"])
    extras = []
    if weapon != "None":
        extras.append(f"A {weapon.lower()} was reportedly used during the commission of the offence.")
    if area_type == "Rural":
        extras.append("The incident occurred in a rural locality with limited street lighting.")
    elif area_type == "Industrial":
        extras.append("The location falls within the industrial zone with minimal civilian presence during the incident time.")
    extras.append(f"The incident was reported at {district} police jurisdiction.")
    return " ".join([base] + extras)

def generate_witness_statement(accused_gender):
    template = random.choice(WITNESS_TEMPLATES)
    times = ["30 minutes", "an hour", "two hours", "a few minutes"]
    genders = ["male", "female"]
    return template.format(time=random.choice(times), gender=accused_gender or random.choice(genders))

def generate_entities_json(district, state, crime_type, weapon, subtype):
    entities = {
        "GPE": [district, state],
        "WEAPON": [weapon] if weapon != "None" else [],
        "CRIME_TYPE": [crime_type, subtype],
        "DATE": [],
        "PERSON": [],
    }
    if random.random() > 0.5:
        entities["PERSON"].append(f"Accused_{random.randint(1000,9999)}")
    return json.dumps(entities)

def urgency_score(severity, crime_type):
    base = {"Low": random.uniform(0.1, 0.35),
            "Medium": random.uniform(0.35, 0.60),
            "High": random.uniform(0.60, 0.80),
            "Critical": random.uniform(0.80, 1.0)}
    score = base.get(severity, 0.5)
    if crime_type in ["Murder", "Kidnapping", "Robbery"]:
        score = min(1.0, score + 0.1)
    return round(score, 3)

def sentiment_score(crime_type, severity):
    if severity == "Critical":
        return round(random.uniform(-1.0, -0.6), 3)
    elif severity == "High":
        return round(random.uniform(-0.7, -0.3), 3)
    elif severity == "Medium":
        return round(random.uniform(-0.4, 0.0), 3)
    else:
        return round(random.uniform(-0.2, 0.2), 3)

# ─── Main Generation ───────────────────────────────────────────────────────────

print(f"Generating {NUM_RECORDS} synthetic crime records...")

records = []
crime_type_list = list(CRIME_TYPES.keys())
weights = [CRIME_TYPES[c]["weight"] for c in crime_type_list]

for i in range(NUM_RECORDS):
    if i % 10000 == 0:
        print(f"  Progress: {i}/{NUM_RECORDS}")

    # Basic identifiers
    report_id = str(uuid.uuid4())[:12].upper()
    report_date = random_date()
    state = random.choice(list(STATES_DISTRICTS.keys()))
    district = random.choice(STATES_DISTRICTS[state])
    lat, lon = generate_lat_lon(state)

    # Crime selection
    crime_type = random.choices(crime_type_list, weights=weights, k=1)[0]
    crime_info = CRIME_TYPES[crime_type]
    subtype = random.choice(crime_info["subtypes"])
    severity = random.choice(crime_info["severity"])
    weapon = random.choice(crime_info["weapons"])
    area_type = random.choice(AREA_TYPES)
    tod = random.choice(TIME_OF_DAY)
    dow = random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    # Persons
    victim_age = random.randint(8, 80)
    victim_gender = random.choice(["Male", "Female", "Other"])
    victim_count = random.choices([1, 2, 3, 4, 5], weights=[0.6, 0.2, 0.1, 0.06, 0.04])[0]
    accused_count = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.25, 0.13, 0.07, 0.05])[0]
    accused_known = random.choices(["Yes", "No", "Partially Known"], weights=[0.35, 0.50, 0.15])[0]
    accused_gender = random.choice(["Male", "Female", "Mixed", "Unknown"])
    accused_age_range = random.choice(["15-25", "25-35", "35-50", "50+", "Unknown"])
    victim_occupation = random.choice(OCCUPATIONS)

    # Narrative
    description = generate_incident_description(crime_type, subtype, area_type, district, weapon)
    witness_stmt = generate_witness_statement(accused_gender)
    officer_remarks = random.choice(OFFICER_REMARKS_TEMPLATES)
    action = random.choice(ACTION_TAKEN)

    # Financial
    prop_stolen = crime_type in ["Theft", "Robbery", "Fraud", "Burglary"]
    estimated_loss = round(random.uniform(500, 5000000), 2) if prop_stolen else 0.0

    # Case metadata
    arrest_made = random.choices(["Yes", "No"], weights=[0.38, 0.62])[0]
    case_resolved = random.choices(["Yes", "No", "Pending"], weights=[0.25, 0.35, 0.40])[0]
    resolution_days = random.randint(1, 730) if case_resolved == "Yes" else None
    evidence_type = random.choice(EVIDENCE_TYPES)
    media_coverage = random.choices(["Yes", "No"], weights=[0.12, 0.88])[0]
    gang_involvement = random.choices(["Yes", "No"], weights=[0.15, 0.85])[0]
    repeat_offence = random.choices(["Yes", "No"], weights=[0.2, 0.8])[0]
    mo = random.choice(MODUS_OPERANDI)

    # Analytics targets
    u_score = urgency_score(severity, crime_type)
    urgency_label = "HIGH" if u_score >= 0.7 else ("MEDIUM" if u_score >= 0.4 else "LOW")
    sent_score = sentiment_score(crime_type, severity)
    entities_json = generate_entities_json(district, state, crime_type, weapon, subtype)

    fir_number = f"FIR/{report_date.year}/{district[:3].upper()}/{str(i+1).zfill(6)}"
    court_case = f"CC/{random.randint(100,9999)}/{report_date.year}" if arrest_made == "Yes" else ""

    records.append({
        # Group 1: Report Identity
        "report_id": report_id,
        "fir_number": fir_number,
        "report_date": report_date.strftime("%Y-%m-%d"),
        "report_time": f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
        "filing_method": random.choice(FILING_METHODS),
        "reporting_officer_id": random.choice(OFFICER_IDS),
        "station_code": random.choice(STATION_CODES),
        "district": district,
        "state": state,
        "status": action,

        # Group 2: Crime Details
        "crime_type": crime_type,
        "crime_subtype": subtype,
        "severity_level": severity,
        "urgency_score": u_score,
        "weapons_used": weapon,
        "modus_operandi": mo,
        "gang_involvement": gang_involvement,
        "repeat_offence": repeat_offence,

        # Group 3: Location
        "incident_location": f"{random.choice(LANDMARKS)}, {district}",
        "area_type": area_type,
        "latitude": lat,
        "longitude": lon,
        "pincode": f"{random.randint(100000, 999999)}",
        "landmark": random.choice(LANDMARKS),
        "beat_number": f"BEAT-{random.randint(1,50):02d}",
        "time_of_day": tod,
        "day_of_week": dow,

        # Group 4: Persons
        "victim_age": victim_age,
        "victim_gender": victim_gender,
        "victim_occupation": victim_occupation,
        "victim_count": victim_count,
        "accused_known": accused_known,
        "accused_count": accused_count,
        "accused_age_range": accused_age_range,
        "accused_gender": accused_gender,

        # Group 5: Narrative Text (NLP Input)
        "incident_description": description,
        "witness_statement": witness_stmt,
        "officer_remarks": officer_remarks,
        "action_taken": action,

        # Group 6: Analytics Labels
        "predicted_crime_type": crime_type,
        "urgency_label": urgency_label,
        "sentiment_score": sent_score,
        "entities_json": entities_json,
        "case_resolved": case_resolved,
        "resolution_days": resolution_days,

        # Group 7: Evidence & Metadata
        "evidence_collected": random.choice(["Yes", "No"]),
        "evidence_type": evidence_type,
        "property_stolen": "Yes" if prop_stolen else "No",
        "estimated_loss_inr": estimated_loss,
        "arrest_made": arrest_made,
        "court_case_number": court_case,
        "media_coverage": media_coverage,
        "data_source": "Synthetic",
    })

print("Building DataFrame...")
df = pd.DataFrame(records)

output_file = "crime_reports_synthetic.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset saved: {output_file}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nCrime type distribution:")
print(df['crime_type'].value_counts())
print(f"\nUrgency label distribution:")
print(df['urgency_label'].value_counts())
print(f"\nSeverity distribution:")
print(df['severity_level'].value_counts())
print(f"\nSample record:")
print(df.iloc[0][['report_id','fir_number','crime_type','severity_level',
                   'urgency_label','district','state']].to_string())
