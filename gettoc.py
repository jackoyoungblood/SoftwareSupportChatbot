import sys, fitz, pickle, os

if not os.path.exists('C:\dev\pdfconversion\IRMWEB_toc.pickle'):
    fname = 'C:\dev\pdfconversion\IRM Web Client User Guide (Legal Version) 10.3.3.pdf'
    doc = fitz.open(fname)  # open document
    out = open(fname + ".txt", "wb")  # open text output
    toclist = doc.get_toc(True)
    with open('C:\dev\pdfconversion\IRMWEB_toc.pickle', 'wb') as handle:
        pickle.dump(toclist, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    toclist = pickle.load(open('C:\dev\pdfconversion\IRMWEB_toc.pickle', 'rb'))

llist = list()

for i in range(len(toclist)):
    toclist[i] = str(toclist[i]).lower()

filteredl = list(filter(lambda a: 'relating' in a, toclist))

pagel = []

for s in filteredl:
    sl = s.split(',')
    pagel.append(int(sl[2].replace(' ','').replace(']','')))


pdfdoc = fitz.open(fname)  # open document
pdfdoc.load_page(pagel[0]) #why isn't intellisense displaying load_page? I installed fitz and pymupf

