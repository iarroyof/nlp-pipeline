# -*- coding: utf-8 -*-
# Script for storing the sparse data into a database. 
# Dependencies: MRjob, psycopg2, postgresql and/or Hadoop.
# Verify if the same task can be done by simpy usong gnu parallel.

# Authors: Julián Solórzano Soto, Ignacio Arroyo-Fernández, Carlos Francisco Méndez Cruz.
import psycopg2
import re
import argparse
from mrjob.job import MRJob

dbname = "fr_dsm_word_space"
dsm = True # activate this flag for parsing a cooccurrence dictionary; disativate it for parsing a word-document list matrix.

def unicodize(segment):
    if re.match(r'\\u[0-9a-f]{4}', segment):
        return segment.decode('unicode-escape')
    return segment.decode('utf-8')

def replaced(item):
    replaced = u"".join((unicodize(seg) for seg in re.split(r'(\\u[0-9a-f]{4})', item)))
    word = replaced.strip('"')
    return word

def insert_list_vector(cursor, word_id, vector):
   inside = False
   number = ""
   pos = 0
   val = 0
   for c in vector:
        if c == '[':
            inside = True
        elif c.isdigit():
                number += c
        elif c == ',':
            if inside:
                pos = int(number)
                number = ""
        elif c == ']':
            if inside:
                val = int(number)
                number = ""
                cursor.execute("insert into word_sparse(word_id, pos, val) values (%s, %s, %s)", (word_id, pos, val))
            inside = False

def insert_dict_vector(cursor, word, vector):
        palabra = word #replaced(palabra)
        d = vector #item[1] 
        bkey = True
        bvalue = False
        key = ""
        value = ""
        for c in d:
            if c == '{':
                pass
            elif c == ":":
                bkey = False
                bvalue = True
            elif c in (",","}"):
                bkey = True
                bvalue = False
                key = replaced(key.strip())
                value = int(value)
                sql = "INSERT INTO coocurrencias VALUES('%s', '%s', %s);"%(palabra, key, value)
                cursor.execute(sql)
                key = ""
                value = ""
            elif bkey:
                key += c
            elif bvalue:
                value += c

def create_tables(cr):
    if dsm:   
        cr.execute("create table if not exists coocurrencias(pal1 character varying, pal2 character varying, valor integer)")
        cr.execute("create table if not exists words(id integer, word character varying)") #(id integer, word character varying, freq integer)
    else:
        cr.execute("create table if not exists word_list(id serial primary key, word character varying not null)")
        cr.execute("""create table if not exists word_sparse(
                  id serial primary key, word_id integer references word_list(id) not null,
                  pos integer not null, val float not null)""")

class MRwordStore(MRJob):

    def mapper_init(self):
        self.conn = psycopg2.connect("dbname="+ dbname +" user=semeval password=semeval")

    def mapper(self, _, line):
        self.cr = self.conn.cursor()
        item = line.strip().split('\t')
        replaced = u"".join((unicodize(seg) for seg in re.split(r'(\\u[0-9a-f]{4})', item[0])))
        key = u''.join((c for c in replaced if c != '"'))     
	if dsm:
	    self.cr.execute("insert into words(word) values(%s) returning id", (key,))
            word_id = self.cr.fetchone()[0]
            insert_dict_vector(cursor = self.cr, word = key, vector = item[1])
        else:	       
            self.cr.execute("insert into word_list(word) values(%s) returning id", (key,))
            word_id = self.cr.fetchone()[0]
            insert_list_vector(cursor = self.cr, word_id = word_id, vector = item[1])

    def mapper_final(self):
        self.conn.commit()
        self.conn.close()

if __name__ == "__main__":
    """Stores word vectors into a database. Such a db (e.g. here is en_ws) must be previusly created in postgresql. 
    It also asumes the owner of the database is a user named semeval with password semeval.
    This script parses input_file.txt containing lines in the next example format (dsm=False):
    
        "word"<tab>	[[number, number],[number, number], ...]
         
        or (dsm=True)

        "word"<tab>	{key:value, key:value, ...}

    Use example:

        python db_store_hadoop.py -r hadoop input_file.txt
    """
    # Firstly create tables once for avoiding duplicates.
    conn = psycopg2.connect("dbname="+ dbname +" user=semeval password=semeval")
    create_tables(conn.cursor())
    conn.commit()
    conn.close()    
    
    # Run the MR object
    MRwordStore().run()
       
