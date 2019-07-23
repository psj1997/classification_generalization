# Work_report2



This week I have tried one transfer learning method called Transfer Kernel Learning. And I try to reproduce this method and use it to do some experiments and the experiment shows that the method can improve the result to some extent.

Firstly I will describe the method briefly. 



### Transfer Kernel Learning Algorithm

This algorithm is proposed in 'Domain Invariant Transfer Kernel Learning'  .

The purpose of this method is to find  a way to reduce the distribution difference between training and the  testing data ,then reducing the generalization error between the training and testing data. This method will find a domain-invariant kernel by matching the training and testing in the reproduced kernel Hilbert space(RKHS). After finding the kernel for the training data and the testing data , then the kernel can be used in kernel machine learning classifiers like kernel SVM or other. I have tried SVM with the learned kernel matrix to train the model and evaluate it in the testing data.



### Code

In the github repo 

TKL.py :Transfer kernel learning , it will return the domain-invariant kernel matrix

classification_TKL.py : It will reproduce the classification result with Study to Study Transfer and Leave One Study Out,including the accuracy matrix,recall matrix and predict_proba matrix.



### Experiments Results

This is the comparison of AUC result between Logistic Regression and kernel SVM with transfer kernel learning(linear kernel).



##### AUC value of stst.

Left is the logistic regression result and the right is the transfer kernel learning with SVM.

<div align=left> 
    <img src="picture7_23\figure_paper_AUC_stst.png" width="350px" >
    <img src="picture7_23\figure_TKL_AUC_stst.png" width="350px">
</div>



From the picture ,we can find that the right results is better than the left in most of the squares. That just a few results in the right is worser than the left .



##### AUC value of loso

This the value of AUC in Leave One Study Out.

Left is the value of paper and right is the value of the transfer kernel learning  SVM. The white column is the average value of other four studies as training set in Study To Study Transfer. And the gray column is the value of the LOSO.

<div align=left> 
    <img src="picture7_23\figure_paper_AUC_loso.png" width="350px" >
    <img src="picture7_23\figure_TKL_AUC_loso.png" width="350px">
</div>

From the picture I can find that Transfer kernel learning has improved the result of white columns and there are not big difference on gray columns. For Study To Study Transfer , it use one study as training set ,and test it on other four studies,but in the LOSO , it uses four studies as training set and the other one as testing set.So in the LOSO,the training set has much more information than in the Study to Study Transfer.Maybe it is the reason that results of LOSO does not improve much.



### Conclusion

Experiments show that using Transfer Kernel Learning method in SVM can help improve the generalization between datasets with different distributions.But there are also a little results show that using this method will make the result worse.So I think I should explore this more in next .











  





