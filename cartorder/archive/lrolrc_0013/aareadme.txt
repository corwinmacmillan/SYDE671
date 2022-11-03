PDS_VERSION_ID          = PDS3
RECORD_TYPE             = STREAM
SPACECRAFT_NAME         = "LUNAR RECONNAISSANCE ORBITER"
INSTRUMENT_NAME	        = "LUNAR RECONNAISSANCE ORBITER CAMERA"
TARGET_NAME             = "MOON"

OBJECT                  = TEXT
   INTERCHANGE_FORMAT     = ASCII
   PUBLICATION_DATE       = 2018-08-29

   NOTE                   = "LUNAR RECONNAISSANCE ORBITER CAMERA (LROC)
                             EXPERIMENT DATA RECORD (EDR)
                             LEVEL 2 Version 1.0"

END_OBJECT              =TEXT
END

LUNAR RECONNAISSANCE ORBITER CAMERA (LROC) EXPERIMENT DATA RECORD (EDR) ARCHIVE

                 Table of Contents

  1. - INTRODUCTION
  2. - FILE FORMATS
  3. - DIRECTORY AND CONTENTS


1. INTRODUCTION


2. FILE FORMATS

All document files and tables are stored as ASCII stream-record files.
In a stream-record file, records (lines of text) are separated by a
carriage-return <cr> and line-feed <lf> character sequence.  The
<cr>/<lf> sequence marks the end-of-record and the start of a new
record.  This organization works well for the Microsoft-DOS systems
because the <cr>/<lf> sequence is identically used on these systems.
On Macintosh systems, an end-of-record mark is simply a <cr>
character.  Macintosh text editors can read and access these files,
but a special-character indicator (usually a "square box" character)
will mark the "extraneous" <lf> character at the beginning of each
line.  On UNIX systems, an end-of-record mark is simply a <lf>
character.  UNIX text editors can read and access these files, but a
special-character (usually a ^M sequence) indicator will mark the
"extraneous" <cr> character at the end of each line.  File names with
extension "TAB", "LBL", "LAB", "TXT", and "CAT" are formatted as ASCII
stream-record files.

Tabular files are formatted so that they may be read directly into
many database management systems of various computers.  All fields are
separated by commas, and character fields are enclosed in double
quotation marks (").  Character fields are left justified, and numeric
fields are right justified.  The "start byte" and "bytes" values
listed in the labels that describe the tabular files do not include
the commas between fields or the quotation marks surrounding character
fields.  The records are of fixed length, and the last two bytes of
each record contain the ASCII <cr>/<lf> characters.  This scheme
allows a table to be treated as a fixed- length record file on
computers that support this file type and as a normal text file on
other computers.


3. DIRECTORY AND CONTENTS

DIRECTORY/FILE          CONTENTS
-------------------     ------------------------------------------

<root>
|
|-AAREADME.TXT          The file you are reading (ASCII Text).
|
|-ERRATA.TXT            Description of known anomalies and errors
|                       present on the volume set (optional file).
|
|-VOLDESC.CAT           A description of the contents of this
|                       release volume in a format readable by
|                       both humans and computers.
|
|-<CALIB>               Directory for the calibration files.
|
|
|-<CATALOG>             Catalog Directory
|  |
|  |-DATASET.CAT        LROC Dataset description.
|  |
|  |-INST.CAT           LROC instrument description.
|  |
|  |-PERSON.CAT         Contributors to this dataset.
|  |
|  |-REF.CAT            Detail on references made in other documents.
|  |
|  |
|-<DOCUMENT>            Documentation Directory.  The files in this
|  |                    directory provide detailed information
|  |                    regarding the LROC EDR archive.
|  |
|  |-DOCINFO.TXT        Description of files in the DOCUMENT
|  |                    directory.
|  |
|  |-LROCDATASIS.TXT    LROC EDR/CDR DATA PRODUCT Software Interface 
|  |			Specification (SIS) document in text format.
|  |
|  |-LROCDATASIS.PDF    LROC EDR/CDR DATA PRODUCT Software Interface     
|  |                    document in PDF format.
|  |
|  |-LROCDATASIS.LBL    PDS label for LROCDATASIS.TXT.
|  |
|  |
|-<INDEX>               Directory for the image index files.
|  |
|  |-INDXINFO.TXT       Description of files in <INDEX> directory.
|  |
|  |-INDEX.TAB          Image Index table detailing EDR contents in <DATA>.
|  |
|  |-INDEX.LBL          PDS label for INDEX.TAB.
|  |
|  |-CUMINDEX.TAB       Cumulative Image Index table for all releases.
|  |
|  |-CUMINDEX.LBL       PDS label for CUMINDEX.TAB.
|  |
|  |
|-<DATA>                Directory for experiment binary data records (NAC/WAC)

