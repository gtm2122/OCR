### Debug code- used to remove every output cell of a jupyter notebook incase the notebook was instructed to print more than 15MB of text
### which causes it to hang.
### Usage in terminal:
### python strip_output.py < overloadednotebook.ipynb > outputless_notebook.ipynb


def strip_output(nb):
 for ws in nb.worksheets:
  for cell in ws.cells:
   if hasattr(cell,"outputs"):
    cell.outputs=[]
   if hasattr(cell,"prompt_number"):
    del cell["prompt_number"]



if __name__=="__main__":
 from sys import stdin,stdout
 from IPython.nbformat.current import read,write

 nb = read(stdin,"ipynb")
 strip_output(nb)
 write(nb,stdout,"ipynb")
 stdout.write('\n')

