# Study Param Questions
Goal: Generate questions related to the study parameters for the study given an association. 

We should have two types of questions:
- Given a correct association, ask the model to identify the p-value and if the result is statistically signficant. The model should also be able to output not found if the p-value is not present in the study. 
- Given an incorrect association (the association is modified in some way similar to what we did in mc questions), ask the model to identify the p-value and if the result is statistically signficant. The model should also be able to output not found if the p-value is not present in the study (this would be the correct answer in this case).