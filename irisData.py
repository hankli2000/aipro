import sklearn
import os

#*** Data
from sklearn import datasets
iris=datasets.load_iris()
print iris.data
print iris.target


#*** start classification 
from sklearn import tree
model=tree.DecisionTreeClassifier()   #initialize model

model.fit(iris.data, iris.target)  #It updates model

#*** Test
X_iris=iris.data[:1, :]  #take the first row 
y_iris=model.predict(X_iris)
print X_iris
print y_iris


"""
#*** Visualize
tree.export_graphviz(model, out_file='iris.dot')
# cmd='dot -Tpdf iris.dot -o iris.pdf'
# os.system(cmd)

from sklearn.tree import export_graphviz
import graphviz
with open('iris.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
"""
