# Full Paper Investigation
- Have the model generate the variants. 
- Recall against ground truth is the starting score
- For each variant, ask all the questions generated that contain that pmcid + variant
- Create a score of percentage of the questions correct from each variant and the final score is the percentage of questions against the recall for each variant so guaranteed to be less than recall
- This would mean the score is against each pmcid and we get a score of how well the model understood the paper