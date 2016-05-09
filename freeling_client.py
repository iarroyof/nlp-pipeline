# -*- coding: utf-8 -*-
import requests

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", help="Input file name (document to be PoS tagged.)", metavar="input_file", required=True)

args = parser.parse_args()

#Archivo a ser enviado
files = {'file': open(args.f, 'rb')}
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
