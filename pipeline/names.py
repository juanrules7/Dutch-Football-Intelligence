"""
One canonical club name per club, whatever the source calls it.

FotMob (Eredivisie), Sofascore (Eerste Divisie) and Transfermarkt all use
different spellings, and Transfermarkt calls the reserve sides "<club> U21"
(Jong Ajax, Jong PSV, Jong AZ, Jong Utrecht).
"""

CANON = {
    # ---- FotMob (Eredivisie) ----
    "ADO Den Haag": "ADO Den Haag", "AZ Alkmaar": "AZ", "Ajax": "Ajax", "Almere City FC": "Almere City",
    "Cambuur": "Cambuur", "Excelsior": "Excelsior", "FC Emmen": "Emmen", "FC Groningen": "Groningen",
    "FC Twente": "Twente", "FC Utrecht": "Utrecht", "FC Volendam": "Volendam", "Feyenoord": "Feyenoord",
    "Fortuna Sittard": "Fortuna Sittard", "Go Ahead Eagles": "Go Ahead Eagles", "Heracles": "Heracles",
    "NAC Breda": "NAC Breda", "NEC Nijmegen": "NEC", "PEC Zwolle": "PEC Zwolle", "PSV Eindhoven": "PSV",
    "RKC Waalwijk": "RKC Waalwijk", "SC Heerenveen": "Heerenveen", "Sparta Rotterdam": "Sparta Rotterdam",
    "Telstar": "Telstar", "Vitesse": "Vitesse", "Willem II": "Willem II",
    # ---- Sofascore (Eerste Divisie) ----
    "De Graafschap": "De Graafschap", "FC Den Bosch": "Den Bosch", "FC Dordrecht": "Dordrecht",
    "FC Eindhoven": "Eindhoven", "Helmond Sport": "Helmond Sport", "Heracles Almelo": "Heracles",
    "Jong AZ Alkmaar": "Jong AZ", "Jong Ajax": "Jong Ajax", "Jong FC Utrecht": "Jong Utrecht",
    "Jong PSV Eindhoven": "Jong PSV", "MVV Maastricht": "MVV", "Roda JC Kerkrade": "Roda JC",
    "SC Cambuur": "Cambuur", "SC Telstar": "Telstar", "TOP Oss": "TOP Oss", "VVV-Venlo": "VVV-Venlo",
    "Willem II Tilburg": "Willem II",
    # ---- Transfermarkt ----
    "Ajax Amsterdam": "Ajax", "AZ Alkmaar U21": "Jong AZ", "Ajax Amsterdam U21": "Jong Ajax",
    "FC Utrecht U21": "Jong Utrecht", "PSV Eindhoven U21": "Jong PSV",
    "De Graafschap Doetinchem": "De Graafschap", "Excelsior Rotterdam": "Excelsior",
    "FC Twente Enschede": "Twente", "Feyenoord Rotterdam": "Feyenoord",
    "SC Cambuur Leeuwarden": "Cambuur", "Vitesse Arnhem": "Vitesse", "MVV Maastricht": "MVV",
}

RESERVE_SIDES = {"Jong Ajax", "Jong PSV", "Jong AZ", "Jong Utrecht"}


def canon(name: str) -> str:
    if name not in CANON:
        raise KeyError(f"Unmapped club name: {name!r} - add it to pipeline/names.py")
    return CANON[name]
