# Version 0.1 Juan Manuel Torres juan-manuel.torres@univ-avignon.fr
# Ce programme est libre
# usage: perl split.perl < TEXTE.txt

#!/usr/bin/perl -w
use strict;
use utf8;
binmode STDOUT, ":encoding(utf8)";
binmode STDIN,  ":encoding(utf8)";

my @string = <>;
my $string = join " ",@string; 	# Coller les morceux dans un string

	    #-------------------------------------------- INEX 2013 ------------
	    $string =~ s/\{\{[^\}]+\}\}/ /g ;		# enlever les {{}}
	    #-------------------------------------------- INEX 2013 ------------

   $string =~ s/\n/µ/g;	  	# preserver les sauts de ligne
   $string =~ s/ +/ /g;		
   $string =~ s/>+/ /g;	  	# eliminer les >
   $string =~ s/-+/ - /g;	# eliminer les -

   $string =~ s/ i\.e\. / i²e² /g;	 # Exception i.e.
   $string =~ s/ U\.S\. / U²S² /g;	 # Exception U.S.
   $string =~ s/ (Jan|Feb|Mar|Apr|Jui|Jul|Aug|Sep|Oct|Nov|Dec)\. / $1² /g;	 # Exception Months


# The sentence boundary algorithm used here is based on one described
# by C. Manning and H. Schutze. 2000. Foundations of Statistical Natural
# Language Processing. MIT Press: 134-135.

# abbreviations that (almost) never occur at the end of a sentence
my @known_abbr = qw/Gen al prof Prof ph d Ph C D H K dr Dr M Mme mr Mr mrs Mrs ms Ms vs vol cit pp ed cap p P R S W cf L N E v chap ch I No St T J F 1o 2o/;

# abbreviations that can occur at the end of sentence
my @sometimes_abbr = qw/etc jr Jr sr Sr/;

my $pbm = '<pbound/>'; # tentative boundary marker

# JMT Let ... as non boundaries
$string =~ s/\.\.+/²²²/g;

# JMT put a tentative sent. boundary marker after \n\n. Important all sentences must to have a space at the end
$string =~ s/µ\s*µ+/ µ$pbm/g;

# put a tentative sent. boundary marker after all .?!
$string =~ s/([.?!])/$1$pbm/g;

# JMT  put a tentative sent. boundary marker before -
$string =~ s/(–)/$pbm $1/g;

# move the boundary after quotation marks
$string =~ s/$pbm"/"$pbm/g;
$string =~ s/$pbm'/'$pbm/g;
$string =~ s/$pbm’/’$pbm/g;
$string =~ s/$pbm»/»$pbm/g;
$string =~ s/$pbm”/”$pbm/g;

# remove boundaries after certain abbreviations
foreach my $abbr (@known_abbr) {
$string =~ s/\b$abbr(\W*)$pbm/$abbr$1 /g;}

foreach my $abbr (@sometimes_abbr) {
$string =~ s/$abbr(\W*)\Q$pbm\E\s*([a-z])/$abbr$1 $2/g;}

# remove . boundaries if followed by uc letter uc letter or number
$string =~ s/([a-z0-9])([.])$pbm([a-z0-9])/$1$2$3/g;

# remove !? boundaries if not followed by uc letter
$string =~ s/([!?])\s*$pbm\s*([a-z])/$1 $2/g;

# JMT remove . boundaries if followed by uc letter and before a uc letter 2 times
$string =~ s/([A-Z]\.)\s*$pbm\s*([A-Z]\.)\s*$pbm\s*([A-Z]\.)/$1$2$3/g;

# JMT remove . boundaries if followed by uc letter and before a uc letter
$string =~ s/([A-Z]\.)\s*$pbm\s*( [A-Z]\.)/$1 $2/g;

# JMT remove . boundaries if followed by ,
$string =~ s/(\.)\s*$pbm\s*(\,)/$1$2/g;

# JMT remove !? boundaries if followed by ) or ]
$string =~ s/([!?])\s*$pbm\s*([\)|\]])/$1$2/g;

# JMT remove !?. boundaries if followed by "  or '
$string =~ s/([.!?])\s*(["'])\s*$pbm/$1$2/g;

# all remaining boundaries are real boundaries
#my @sentences = map {s/^\s+|\s+$//g; $_} split /[.?!]+\Q$pbm\E/, $string;

# JM Remettre les .
$string =~ s/²/./g;

# JM Eliminer les µ
$string =~ s/µ//g;

# JM Couper en $pbm, garder le symbole de ponctuation
$string =~ s/$pbm/\n/g; $string =~ s/^\s+|\s+$/ /g;
@string=split/\n/,$string;

my $num=-1;
foreach my $sent (@string) {
	next if $sent =~ /^\s*$/;
	next if $sent =~ /^\s*\.$/;
	next if $sent =~ /^\.$/;
	$sent =~ s/</(/g;	# pour compatibilite XML
	$sent =~ s/>/)/g;	# pour compatibilite XML
	$sent = " ".$sent;
	$sent =~ s/\s+/ /g;
#	print $num++."\t$sent\n";	# numbers
	print "$sent\n";
}

