import requests

url = "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-full_R3.01.txt"  # Sostituisci con l'URL del file .dat da scaricare
response = requests.get(url)

# Specifica il percorso della cartella in cui desideri salvare il file .dat
output_path = "/home/git/AstroCosmoComputational/Students/Gabriele_Russo/CosmicStructures/Week6/dataCMB.dat"  # Sostituisci con il percorso e il nome del file .dat

# Scrive il contenuto del file .dat
with open(output_path, 'wb') as file:
    file.write(response.content)

print(f"File .dat scaricato e salvato in {output_path}")
