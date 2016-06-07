# -*- coding: utf-8 -*-
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="input file name", metavar="input_file", required=True)
#parser.add_argument("-F", help="output file name", metavar="output_file", required=True)

args = parser.parse_args()
#Archivo a ser enviado
files = {'file': open('/home/iarroyof/sushi_fr.txt', 'rb')}
#files['file_'+str(i)] = open('/home/iarroyof/sushi_fr.txt', 'rb')
#Parámetros posibles
#outf: tagged, tagged_en, tagged_fr
#format: json, plain, html

params = {'outf': 'tagged_fr', 'format': 'plain', 'flush':'yes'}
#Enviar petición
url = "http://www.corpus.unam.mx/servicio-freeling/analyze.php"
#for files in Files:
r = requests.post(url, files=files, params=params)
r.encoding  = "utf-8"
#Imprimir respuesta, de aquí se puede guardar en otro archivo
print r.text
#with open(args.F, "w") as f:
#    f.write(r.text)
