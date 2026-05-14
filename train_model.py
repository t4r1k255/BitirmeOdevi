import os
os.makedirs("models", exist_ok=True)

from BitirmeOdevi.SkorlamaModeli import modeli_egit_ve_kaydet

if __name__ == "__main__":
    modeli_egit_ve_kaydet()
