from dotenv import load_dotenv
import os
from asynctinydb import TinyDB, Modifier

if __name__ == "__main__":
    load_dotenv("./config/.env")
    personality_db = TinyDB("./db/personality_db.json")
    Modifier.Conversion.ExtendedJSON(personality_db)

    

