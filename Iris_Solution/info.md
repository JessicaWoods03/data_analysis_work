<h2>Iris Data Set Solution:</h2>

I think Decision Scores were a great insight on how the different ensemble
models worked with the classification processes. Random Forest Classifier
seems to the be the best bet, or a meta stacked version with a decision tree. 
All the classification models where promising, even the ensemble models, 
like LightGBM and Boosters, Gradient models, but the Random Forest Classifier
has the best decisions.
I worked with test, train, and validation data for the models. Large datasets are
not really required for some of the basic models I worked with. The meta example
did need large data to work properly though, that was interesting. 
I watched the performance falter as it ran out of data. I did want to play the
different ensemble methods with a decision tree. I did a confusion matrix with the
Classifiers that I stacked using StackClassifier(). The visual shows great promise
with this neat approach, but the LightGBM continued to quit working because it needs 
larger amounts of data to perform better.
I thought the heatmap for the QDA was also fascinating. This heatmap can help me 
understand how the QDA model assigns scores to different instances and how well-separated
the decision regions are. Blue Cells (Low Scores): Instances with blue cells indicate that
the model assigned relatively low decision function scores for all classes. This suggests that
the model is not confident in assigning any specific class to these instances. 
Yellow Cells (High Scores): Instances with yellow cells indicate that the model assigned
higher decision function scores for one or more classes. This suggests that the model is
more confident in assigning these instances to specific classes. Since blue on 0 and yellow on 500, 
it's likely that the decision function scores are highly positive for those instances, indicating
strong confidence in the class assignment. In the heat map Setosa was the only one with strong 
confidence in the classifications of the QDA model. The rest was close to zero.
This gave the best parameters for Best Random Forest Model, which was pretty awesome. I think that
is the best solution to the iris data set. The decision scores also seem to support that. When I looked
at the pairplots, they scored 1 mostly with all the species, the classes are well separated from each
other, in the decision process. 
Best Random Forest Accuracy: 1.0
Best Parameters: {'max_depth': None, 'max_features': 'auto', 
'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
I took what I learned from the Mushroom data 
and worked with the iris data to find more interesting concepts and how those models work. 
Naive Bayes classifiers typically works better with discrete data or count data, so using them with the 
Iris dataset (continuous features) might not provide the best results. I did a class probability visualization
to see how well it will classify the data, it was lacking that ability. Honestly, to see that you need to
visualize what it does with Classes, and what those probabilities are. F1’s and Accuracy Scores, 
ROC (one to all), don’t necessarily give you any insight on how to solve the iris data set. 
You really need to see what the models are doing with the data, and how the model makes those
decisions to predict the outcome.<br>

This was my quiet protest:<br>

qda_res = {}

qda = QuadraticDiscriminantAnalysis().fit(X, y)
y_train_pred = qda.predict(X)

qda_res['accuracy'] = accuracy_score(y, y_train_pred)
qda_res['mcc'] = matthews_corrcoef(y, y_train_pred)
qda_res['f1'] = f1_score(y, y_train_pred, average='weighted')
qda_res

'accuracy': 0.98, 'mcc': 0.970064673134052, 'f1': 0.9799979997999799

<br><b>This does not tell what the model is doing, nor does it tell you if it can do
the job even though the results are reasonable. So I began validating and cross-validating this model.</b><br>There were five of these models written out in the same fashion, none those examples show you why
those would work to solve it. It’s just numbers. My work was deleted by a teammate without explainations, I have no ideal 
who did that?
<b><h3> Personal note on teamwork:</b><br></h3>
I don’t think its right to work against someone or argue their logic. It’s not team work and it
gets you nowhere. I forked over a copy of the Iris Code we were working on and continued my work
that I usually do, which is clean up, and validate the code a fellow student lays down that is missing a lot of
important parameters, like testing, training and validation data, visuals. Four important variables for working
with Machine Learning Models. My job in this team was to come in after
everyone was done and cleaned it up and modeled it out correctly with visuals, commentary, validations. 
I don’t mind doing that, that was my job, like a fellow student was writing and did his part.
I just can’t really move forward when my work gets deleted. I forked over and continued working on my
quiet protest. Never delete a teammates work, always try to validate, or cross validate. 
Otherwise, it is considered rude. Soft Skill lesson to be learned.<br>

<b>In this GITHUB publication, most of that students work has been deleted, save what few items is commented 
to be theirs.</b>

I already knew the answer to the iris data set before I did this, just from exploring binary classifications
with the mushroom data. I had the option to play with more models listed by a fellow student in his brilliant
technical write ups, but I didn’t need to. I feel I provided enough different concepts, models, and visuals to 
understand why the models I chose as the answer is the right answer. In addition to the analysis on P-values, even 
though that was not a requirement or requested in the write-ups. 

There are SVM’s and Regression’s as well but I don’t think I need those explored. 
There is actually two solutions to the data set,
a modified Ensembled Decision Tree or a Random Forest Classifier with the confirmation of hyperparameters.

