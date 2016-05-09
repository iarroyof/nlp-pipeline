# -*- coding: utf-8 -*-
import requests
#Archivo a ser enviado
files = {'file': open('sushi_fr.txt', 'rb')}
#Parámetros posibles
#outf: tagged, tagged_en, tagged_fr
#format: json, plain, html
params = {'outf': 'tagged_fr', 'format': 'plain'}
#Enviar petición
url = "http://www.corpus.unam.mx/servicio-freeling/analyze.php"
r = requests.post(url, files=files, params=params)
r.encoding  = "utf-8"
#Imprimir respuesta, de aquí se puede guardar en otro archivo
print r.text
