
25

Automatic Zoom
1 3Genetic Programming and Evolvable Machines           (2024) 25:10  Page 17 of 27    10 
6.1  Datasets
This  section  describes  the  datasets  used  for  this  research.  These  datasets  were  
chosen  because  the  state-of-the-art  approaches  mentioned  in  Sect.  7.3  make  use  
of these datasets, allowing for a comparison to be made between this research and 
state-of-the-art  methods.  Additionally,  they  contain  a  large  number  of  samples  
which can be adequately split into training and testing sets. The large number of 
samples will help to reduce neural networks from overfitting. The following data-
sets were used to test the proposed approaches for image classification:
• CIFAR-10 - This dataset contains 60,000 32x32 colour images. There are 10 
classes. Each class contains 6000 images. 50,000 images are used for training 
and the remaining 10,000 for testing. [20]
• Fashion  MNIST  -  A  dataset  containing  images  of  various  clothing  items.  
There are 70,000 28x28 greyscale images in total, with 60,000 being used for 
training and 10,000 for testing. There are 10 classes. [49]
• Street  View  House  Numbers  (SVHN)  -  This  dataset  consists  of  600,000  
32x32  RGB  images  of  digits  on  house  number  plating.  Of  these  images,  
73,257 are used for training and 26,032 are for testing with 10 classes. [34]
• EuroSat  -  A  data  set  containing  satellite  images.  In  total  there  are  27,000  
images and 10 classes. [34]
The following datasets were used for video shorts creation:
• PHD2: Personalized Highlight Detection - This dataset contains a selection 
of  various  YouTube  video  IDs,  and  the  timestamp  of  the  highlight  of  each  
video,  along  with  some  user-specific  information.  There  are  12,972  training  
examples and 850 testing examples. [31]
• Video2Gif - A dataset containing 100,000 gifs along with the respective vid-
eos and related data which the gifs were extracted from. [12]
• YouTube-8  M  Segments  Dataset  -  This  dataset  contains  semgments  date  
from  YouTube  videos,  of  which  there  are  about  237,000  segments  on  1000  
classes. There are 5 segments per video. [1]
• QVHighlights  -  This  dataset  consists  of  over  10,000  YouTube  videos,  each  
with annotated highlight information. [22]
6.2  Performance metrics
The  following  metrics  will  be  used  to  evaluate  the  performance  of  the  GPNND  
system:
• Model Accuracy - A measure of the correctness of the predictions of the neu-
ral network, represented as a decimal or percentage.
 Genetic Programming and Evolvable Machines           (2024) 25:10 1 3   10  Page 18 of 27
• Design  Time  -  The  time  taken  for  the  GPNND  system  to  evolve  a  suitable  
architecture for the given dataset. This is measured in hours.
6.3  Parameters
The determination of parameter values for the GPNND and ISBGP approaches were 
guided by a review of related work [2, 4, 13], wherein initial values from previous 
studies were adopted as a foundational starting point. Subsequently, an approach was 
taken for each parameter, involving the exploration of values within ranges slightly 
smaller and larger than the initial settings. Through experimentation and testing, the 
algorithm’s  performance  was  evaluated  across  various  parameter  values.  The  deci-
sion to use the current parameter values emerged from this iterative process, as they 
demonstrated  the  best  results  for  this  research.  A  table  detailing  all  the  parameters  
and their justification is given in Table 4.
6.4  Technical specification
GA,  GP  and  ISBGP-II  are  designed  using  Python  3.  The  TensorFlow,  Keras  and  
DEAP  libraries  are  used.  The  experiments  were  evaluated  on  the  Google  Colab1 
suite, making use of a GPU runtime.
Table 4  Parameters
Name Value Justification
Population size 15 Allows for enough genetic variation and reasonable execution times
Minimum Tree Depth 2 This depth is enough for a minimal working tree
Maximum Tree Depth 20 Allows for ample room for trees to grow and promote variation 
without memory and computational issues
Number of Generations 25 Allows for enough time for the programs to sufficiently evolve
Number of NN Epochs 30 Allows for models to train sufficiently without overfitting
Crossover Probability 0.35 Promotes sufficient genetic variation and genetic collaboration
Mutation Probability 0.45 Introduces sufficient genetic variation without losing good changes
Cut-off depth 4 This depth is before which there are the most similarities between 
trees
Global index threshold 6 Allows for ISBGP to move the search space effectively for global 
areas
Local index threshold value 8 Allows for ISBGP to move the search space effectively for local 
areas
1  https://colab.research.google.com/
1 3Genetic Programming and Evolvable Machines           (2024) 25:10  Page 19 of 27    10 
7  Results
This  section  discusses  the  performance  of  the  GA,  GPNND  and  ISBGP-II  
approaches.  For  each  of  these  approaches,  30  runs  were  performed  for  each  
dataset.  Average  accuracy  and  design  times  were  collected  and  P  values  were  
calculated. The following sections document the results of the approaches.
Table 5  Table of results for image classification
Approach Metrics Dataset
Fashion MNIST CIFAR-10 SVHN EuroSat
GPNND Best Design Time (hrs)  4.6 4.7 4.6 5.3
Best Accuracy  0.9602 0.9531 0.9648 0.9296
Average Accuracy
(Train)
0.9774 0.9693 0.9785 0.9382
Average Accuracy
(Test)
0.9517 0.9396 0.9514 0.9127
GA Best Design Time (hrs) 5.2 5.6 5.6 5.7
Best Accuracy 0.9429 0.8916 0.9098 0.9106
Average Accuracy
(Train)
0.9619 0.9086 0.9191 0.9286
Average Accuracy
(Test)
0.9351 0.8831 0.8896 0.8963
Table 6  Table of results for video short creation
Approach Metrics Dataset
PHD2 Video2GIF 8 M QVHighlights
GPNND Best Design Time (hrs) 9.1 9.6 9.2 9.3
Best Accuracy 0.8912 0.8783 0.8707 0.8376
Average Accuracy
(Train)
0.9105 0.8914 0.8973 0.8649
Average Accuracy
(Test)
0.8827 0.8679 0.8596 0.8297
GA Best Design Time (hrs) 10.8 10.7 10.4 9.8
Best Accuracy 0.7376 0.6891 0.7719 0.8202
Average Accuracy
(Train)
0.7587 0.7114 0.7936 0.8449
Average Accuracy
(Test)
0.7279 0.6825 0.7604 0.8164
 Genetic Programming and Evolvable Machines           (2024) 25:10 1 3   10  Page 20 of 27
7.1  GPNND and GA approaches
The  performance  comparison  for  GPNND  and  GA  for  image  classification  and  
video shorts creation is documented in Tables 5 and 6 respectively.
A  statistical  comparison  between  GPNND  and  GA  was  performed.  The  null  
hypothesis  states  that  the  accuracy  of  GPNND  is  equal  to  that  of  GA  and  the  
design  time  of  GPNND  is  equal  to  that  of  GA.  The  alternative  hypothesis  states  
that GPNND has a higher accuracy and lower design time than GA. The hypoth-
eses  were  tested  with  a  Welch’s  t-test,  with  a  significance  level  of  0.05.  Table  7 
shows  the  P  values  for  the  accuracy  and  design  time  for  this  comparison.  In  
the  table,  all  P  values  are  less  than  the  significance  level  of  0.05  implying  that  
the  null  hypothesis  can  be  rejected  in  favour  of  the  alternative  hypothesis,  thus  
GPNND has higher accuracy and lower design time than GA.
Table 7  P values Data set P value
Accuracy Design time
Fashion MNIST (Image) 0.0435 0.0163
CIFAR-10 (Image) 0.0232 0.0388
SVHN (Image) 0.0164 0.0249
EuroSAT (Image) 0.0242 0.0311
PHD2 (Video) 0.0329 0.0302
Video2GIF (Video) 0.0337 0.0226
QVHigilights (Video)  0.0253 0.0219
YouTube 8 M (Video) 0.0104 0.0168
Table 8  Table of results for image classification
Approach Metrics Dataset
Fashion MNIST CIFAR-10 SVHN EuroSat
GPNND Best design time (hrs) 4.6 4.7 4.6 5.3
Best accuracy 0.9602 0.9531 0.9648 0.9296
Average accuracy
(Train)
0.9774 0.9693 0.9785 0.9382
Average accuracy
(Test)
0.9517 0.9396 0.9514 0.9127
ISBGP-II Best design time (hrs) 4.2 4.2 5.1 5.1
Best accuracy 0.9775 0.9786 0.9815 0.9549
Average accuracy
(Train)
0.9847 0.9863 0.9885 0.9692
Average accuracy
(Test)
0.9637 0.9615 0.9753 0.9395
1 3Genetic Programming and Evolvable Machines           (2024) 25:10  Page 21 of 27    10 
7.2  ISBGP‑II and GPNND approaches
The performance comparison for the ISBGP-II approach and GPNND for 
image  classification  and  video  shorts  creation  is  documented  in  Tables  8  and  9 
respectively.
A  statistical  comparison  between  ISBGP-II  and  GPNND  was  also  performed.  
The  null  hypothesis  states  that  the  accuracy  of  ISBGP-II  is  equal  to  of  GPNND  
and  the  design  time  of  ISBGP-II  is  equal  to  that  of  GPNND.  The  alternative  
hypothesis states that ISBGP-II has a higher accuracy and lower design time than 
GPNND.  The  hypotheses  were  also  tested  with  a  Welch’s  t-test,  with  a  signifi-
cance level of 0.05. Table 10 shows the P values for the accuracy and design time 
for this comparison. In the table, all P values are less than the significance level 
of 0.05 implying that the null hypothesis can be rejected in favour of the alterna-
tive hypothesis, thus ISBGP-II has a higher accuracy and lower design time than 
GPNND.
Table 9  Table of results for video short creation
Approach Metrics Dataset
PHD2 Video2GIF 8 M QVHighlights
GPNND Best design time 9.1 9.6 9.2 9.3
Best accuracy 0.8912 0.8783 0.8707 0.8376
Average accuracy 
(Train)
0.9105 0.8914 0.8973 0.8649
Average accuracy 
(Test)
0.8827 0.8679 0.8596 0.8297
ISBGP-II Design time 9 9.2 9.4 9.0
Accuracy 0.9167 0.8949 0.8985 0.8682
Average accuracy 
(Train)
0.9264 0.9162 0.9095 0.8892
Average accuracy (Test) 0.9016 0.8877 0.8894 0.8573
Table 10  P values Data set P value
Accuracy Design time
Fashion MNIST (Image) 0.0435 0.0163
CIFAR-10 (Image) 0.0232 0.0388
SVHN (Image) 0.0164 0.0249
EuroSAT (Image) 0.0242 0.0311
PHD2 (Video) 0.0329 0.0302
Video2GIF (Video) 0.0337 0.0226
QVHigilights (Video)  0.0253 0.0219
YouTube 8 M (Video) 0.0104 0.0168
 Genetic Programming and Evolvable Machines           (2024) 25:10 1 3   10  Page 22 of 27
7.3  Comparison with state of the Art
This  section  compares  the  performance  of  GPNND  and  ISBGP-II  with  state  
of  the  art  approaches  applied  to  the  same  datasets.  Please  note  that  this  is  only  
included for the sake of completeness. The aim of the research is not to obtain a 
performance improvement over the state of the art approaches but rather to inves-
tigate the benefit of using GP rather than GA for NAS. The comparison is shown 
in Table 11.
7.4  Evolved designs
This  section  showcases  evolved  designs  which  the  GA,  GPNND  and  ISBGP-II  
approaches  produce  for  the  image  classification  and  video  shorts  creation.  The  
figures  depict  a  graphic  representation  of  the  evolved  neural  networks  where  
each  block  represents  a  layer  of  the  neural  network.  The  GP  trees  produced  and  
evolved  in  practice  grow  to  be  too  large  to  plot  effectively,  thus  making  them  
not interpretable. The trees are therefore treated as a black box, and future work 
will investigate methods for making the classifiers more readable. The final neu-
ral  network  architectures,  which  are  created  by  evaluating  the  evolved  trees  are  
therefore shown.
7.4.1  Evolved designs for image classification
Figure 7 shows the architecture of three neural networks, each evolved using the 
GA, GPNND and ISBGP approaches respectively for image classification:
In  Fig.  7,  graphic  representations  of  three  neural  networks  for  image  classi-
fication  are  shown.  These  are  labelled  ’GA’,  ’GP’  and  ’ISBGP’,  referring  to  the  
three  approaches.  The  best  performing  neural  network  is  evolved  by  ISBGP  and  
achieves an accuracy of 97.75%.
Table 11  Comparison to state 
of the art methods for image 
classification
Approach Dataset
Fashion MNIST CIFAR-10 SVHN
GPNND 0.9602 0.9531 0.9646
ISBGP 0.9775 0.9786 0.9815
Other Methods 0.9691 0.9891 0.9901
[44] [45] [7]
0.9641 0.972 0.99
[7] [50] [36]
0.9444 0.0653 0.989
[9] [23] [24]
1 3Genetic Programming and Evolvable Machines           (2024) 25:10  Page 23 of 27    10 
7.4.2  Evolved designs for video short creation
Figure 8 shows the architecture of three neural networks, each evolved using the 
GA, GPNND and ISBGP approaches respectively for video shorts creation:
Fig. 7  Neural network architectures evolved by GA, GP and ISBGP-II for Image Classification
Fig. 8  Neural network architectures evolved by GA, GP and ISBGP-II for Video Shorts Creation
 Genetic Programming and Evolvable Machines           (2024) 25:10 1 3   10  Page 24 of 27
In Fig. 8, graphic representations of three neural networks for video shorts crea-
tion  are  shown.  These  are  labelled  ’GA’,  ’GP’  and  ’ISBGP’,  referring  to  the  three  
approaches. The best performing neural network is evolved by ISBGP and achieves 
an accuracy of 91.07%.
For  both  image  classification  and  video  shorts  creation,  the  neural  netowrk  for  
GPNND  contains  fewer  layers  than  the  neural  network  for  GA,  while  achieving  a  
higher accuracy. The same can be seen for ISBGP-II where the neural network con-
tains  fewer  layers  but  achieves  a  higher  accuracy.  The  layer  parameter  values  are  
also  different  in  the  approaches.  GPNND  and  ISBGP-II  use  different  numbers  of  
convolutions and kernel sizes for pooling in their respective layers.
8  Conclusions and future work
The main aim of the research presented in this paper was to study the use of genetic 
programming for NAS. Both canonical GP (GPNND) and a variation of GP which 
takes both structure and fitness into consideration when directing the search, ISBGP-
II, were examined for NAS for image processing and video shorts creation. The per-
formance of GPNND and ISBGP-II was also compared to a genetic algorithm (GA) 
for  NAS.  Both  GPNND  and  ISBGP-II  outperformed  the  GA  for  NAS  for  image  
classification and video shorts creation. ISBGP-II was found to perform better than 
GPNND as well as a previous version of the approach.
Given  the  effectiveness  of  transfer  learning  in  GP  [39],  future  work  will  exam-
ine  the  use  of  transfer  learning  in  ISBGP-II  for  NAS.  In  addition  to  this,  fitness  
approximation  techniques  will  also  be  examined  to  reduce  the  computational  cost  
of ISBGP-II. Future work will also investigate combination operators that will form 
part of the GP function set to combine layers in a neural network architecture.
Author Contributions  R. Kapoor wrote the main manuscript. N. Pillay acted as supervisor and reviewed 
the manuscript.
Funding   Open  access  funding  provided  by  University  of  Pretoria.  This  work  was  funded  as  part  of  the  
Multichoice Research Chair in Machine Learning at the University of Pretoria, South Africa. This work 
is based on the research supported in part by the National Research Foundation of South Africa (Grant 
Number 138150). Opinions expressed and conclusions arrived at, are those of the author and are not nec-
essarily to be attributed to the NRF.
Declarations 
Conflict of interest  The authors have no competing interests as defined by Springer, or other interests that 
might be perceived to influence the results and/or discussion reported in this paper.
Open  Access   This  article  is  licensed  under  a  Creative  Commons  Attribution  4.0  International  License,  
which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long 
as  you  give  appropriate  credit  to  the  original  author(s)  and  the  source,  provide  a  link  to  the  Creative  
Commons  licence,  and  indicate  if  changes  were  made.  The  images  or  other  third  party  material  in  this  
article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line 
to the material. If material is not included in the article’s Creative Commons licence and your intended 
1 3Genetic Programming and Evolvable Machines           (2024) 25:10  Page 25 of 27    10 
use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permis-
sion directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/
licenses/by/4.0/.
References   1.  S.  Abu-El-Haija,  N.  Kothari,  J.  Lee,  et  al,  Youtube-8m:  A  large-scale  video  classification  bench-
mark. CoRR abs/1609.08675. http:// arxiv. org/ abs/ 1609. 08675, (2016)  2. Y. Bi, B. Xue, M. Zhang, An evolutionary deep learning approach using genetic programming with 
convolution operators for image classification. In: 2019 IEEE Congress on Evolutionary Computa-
tion (CEC), pp 3197–3204 (2019). https:// doi. org/ 10. 1109/ CEC. 2019. 87901 51   3.  P.  Covington,  J.  Adams,  E.  Sargin,  Deep  neural  networks  for  youtube  recommendations.  In:  Pro-
ceedings  of  the  10th  ACM  Conference  on  Recommender  Systems.  Association  for  Computing  
Machinery, New York, NY, USA, RecSys ’16, pp 191–198 (2016). https:// doi. org/ 10. 1145/ 29591 00. 29591 90  4. S. Deng, Y. Sun, E. Galvan, Neural architecture search using genetic algorithm for facial expression 
recognition. In: Proceedings of the Genetic and Evolutionary Computation Conference Companion. 
Association  for  Computing  Machinery,  Boston,  Massachusetts,  GECCO  ’22,  pp  423–426  (2022).  
https:// doi. org/ 10. 1145/ 35203 04. 35288 84   5.  J.  Diniz,  F.  Cordeiro,  P.  Miranda,  et  al,  A  grammar-based  genetic  programming  approach  to  opti-
mize  convolutional  neural  network  architectures.  In:  Proceedings  of  the  XV  National  Meeting  of  
Artificial and Computational Intelligence, https:// doi. org/ 10. 5753/ eniac. 2018. 4406 (2018)  6. T. Elsken, J.H. Metzen, F. Hutter, Neural architecture search: a survey. J. Mach. Learn. Res. 20(1), 
1997–2017 (2019)  7. P. Foret, A. Kleiner, H. Mobahi, et al, Sharpness-aware minimization for efficiently improving gen-
eralization.  In:  International  Conference  on  Learning  Representations,  (2021)  https:// openr eview. net/ forum? id= 6Tm1m poslrM  8. G. Franchini, V. Ruggiero, F. Porta et al., Neural architecture search via standard machine learning 
methodologies.  Math.  Eng.  5(1),  1–21  (2022).  https:// doi. org/ 10. 3934/ mine. 20230 12www. aimsp ress. com/ artic le/ doi/ 10. 3934/ mine. 20230 12   9.  P.  Gavrikov,  J.  Keuper,  Cnn  filter  db:  An  empirical  investigation  of  trained  convolutional  filters.  
In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 19,044–
19,054 (2022). https:// doi. org/ 10. 1109/ CVPR5 2688. 2022. 01848 10. I.J. Goodfellow, Y. Bengio, A. Courville, Deep learning (MIT Press, Cambridge, MA, USA, 2016) 11. Z. Guo, X. Zhang, H. Mu et al., Single path one-shot neural architecture search with uniform sam-
pling,  in  Computer  Vision  -  ECCV  2020.  ed.  by  A.  Vedaldi,  H.  Bischof,  T.  Brox  et  al.  (Springer  
International Publishing, Cham, 2020), pp.544–560 12. M. Gygli, Y. Song, L. Cao, Video2gif: Automatic generation of animated gifs from video. pp 1001–
1009 (2016). https:// doi. org/ 10. 1109/ CVPR. 2016. 114 13. T. Hassanzadeh, D. Essam, R. Sarker, Evodcnn: an evolutionary deep convolutional neural network 
for image classification. Neurocomputing (2022). https:// doi. org/ 10. 1016/j. neucom. 2022. 02. 003 14. K. He, X. Zhang, S. Ren, et al, Deep residual learning for image recognition. 1512.03385 (2015) 15. Y. Jaafra, J. Luc Laurent, A. Deruyver et al., Reinforcement learning for neural architecture search: 
A review. Image Vis. Comput. 89, 57–66 (2019). https:// doi. org/ 10. 1016/j. imavis. 2019. 06. 005www. scien cedir ect. com/ scien ce/ artic le/ pii/ S0262 88561 93008 85  16.  H.  Jiang,  Y.  Lu,  J.  Xue,  Automatic  soccer  video  event  detection  based  on  a  deep  neural  network  
combined cnn and rnn. In: 2016 IEEE 28th International Conference on Tools with Artificial Intel-
ligence (ICTAI), pp 490–494 (2016). https:// doi. org/ 10. 1109/ ICTAI. 2016. 0081 17. R. Kapoor, N. Pillay, Iterative structure-based genetic programming for neural architecture search. 
In: Proceedings of the Companion Conference on Genetic and Evolutionary Computation. Associa-
tion for Computing Machinery, New York, NY, USA, GECCO ’23 Companion, pp 595–598 (2023). 
https:// doi. org/ 10. 1145/ 35831 33. 35907 59 18. A. Klos, M. Rosenbaum, W. Schiffmann, Neural architecture search based on genetic algorithm and 
deployed in a bare-metal kubernetes cluster. Int. J. Netw. Comput. 12(1), 164–187 (2022)
 Genetic Programming and Evolvable Machines           (2024) 25:10 1 3   10  Page 26 of 27  19.  J.R.  Koza,  R.  Poli,  Genetic  programming,  in  Search  methodologies.  (Springer,  Berlin,  2005),  
pp.127–164 20. A. Krizhevsky, Learning multiple layers of features from tiny images. Tech. Rep (2009) 21. A. Krizhevsky, I. Sutskever, G. Hinton, Imagenet classification with deep convolutional neural net-
works. Neural Inf. Process. Syst. (2012). https:// doi. org/ 10. 1145/ 30653 86 22. J. Lei, T.L. Berg, M. Bansal, Qvhighlights: Detecting moments and highlights in videos via natural 
language queries. (2021). https:// doi. org/ 10. 48550/ ARXIV. 2107. 09609,  23.  X.  Li,  W.  Wang,  X.  Hu,  et  al,  Selective  kernel  networks.  pp  510–519,  https:// doi. org/ 10. 1109/ CVPR. 2019. 00060 (2019) 24. S. Lim, I. Kim, T. Kim, et al, Fast autoaugment (2019) 25. R. Lima, D. Magalhães, A. Pozo et al., A grammar-based gp approach applied to the design of deep 
neural networks. Genet Program Evol Mach (2022). https:// doi. org/ 10. 1007/ s10710- 022- 09432-0  26.  S.  Litzinger,  A.  Klos,  W.  Schiffmann,  Compute-efficient  neural  network  architecture  optimization  
by a genetic algorithm, in Artificial Neural Networks and Machine Learning - ICANN 2019: Deep 
Learning. ed. by I.V. Tetko, V. Kůrková, P. Karpov et al. (Springer International Publishing, Cham, 
2019), pp.387–392 27. Y. Liu, Y. Sun, B. Xue et al., A survey on evolutionary neural architecture search. IEEE Trans Neu-
ral Netw Learn Syst (2021). https:// doi. org/ 10. 1109/ TNNLS. 2021. 31005 54 28. Y. Liu, Y. Sun, B. Xue et al., A survey on evolutionary neural architecture search. IEEE Trans. Neu-
ral Netw. Learn. Syst. 34(2), 550–570 (2023). https:// doi. org/ 10. 1109/ TNNLS. 2021. 31005 54 29. A. McGhie, B. Xue, M. Zhang, Gpcnn: evolving convolutional neural networks using genetic pro-
gramming. In: 2020 IEEE Symposium Series on Computational Intelligence (SSCI), pp 2684–2691 
(2020). https:// doi. org/ 10. 1109/ SSCI4 7803. 2020. 93083 90 30. J.F.  Miller, Cartesian  Genetic  Programming  (Springer,  Berlin  Heidelberg,  Berlin,  Heidelberg,  
2011), pp.17–34. https:// doi. org/ 10. 1007/ 978-3- 642- 17310-3_2  31.  A.  Garcia  del  Molino,  M.  Gygli,  PHD-GIFs:  personalized  highlight  detection  for  automatic  GIF  
creation. In: Proceedings of the 2018 ACM on Multimedia Conference. ACM, New York, NY, USA, 
MM ’18 (2018) 32. K. Muhammad, T. Hussain, M. Tanveer et al., Cost-effective video summarization using deep cnn 
with hierarchical weighted fusion for iot surveillance networks. IEEE Internet Things J. 7(5), 4455–
4463 (2020). https:// doi. org/ 10. 1109/ JIOT. 2019. 29504 69 33. M.H.T. Najaran, A genetic programming-based convolutional deep learning algorithm for identify-
ing  covid-19  cases  via  x-ray  images.  Artif.  Intell.  Med.  142(102),  571  (2023).  https:// doi. org/ 10. 1016/j. artmed. 2023. 102571www. scien cedir ect. com/ scien ce/ artic le/ pii/ S0933 36572 30008 54  34.  Y.  Netzer,  T.  Wang,  A.  Coates,  et  al,  Reading  digits  in  natural  images  with  unsupervised  feature  
learning. In: NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011 (2011). 
http:// ufldl. stanf ord. edu/ house numbe rs/ nips2 011_ house numbe rs. pdf 35. M. O’ Neill, C. Ryan, Grammatica Evolution Evolutionary Automatic Programming in an Arbitrary 
Language. Springer (2003)  36.  N.H.  Phong,  B.  Ribeiro,  Rethinking  recurrent  neural  networks  and  other  improvements  for  image  
classification (2020)  37.  E.  Real,  S.  Moore,  A.  Selle,  et  al,  Large-scale  evolution  of  image  classifiers.  In:  Precup  D,  Teh  
YW (eds) Proceedings of the 34th International Conference on Machine Learning, Proceedings of 
Machine  Learning  Research,  vol  70.  PMLR,  pp  2902–2911  (2017).  https:// proce edings. mlr. press/ v70/ real1 7a. html  38.  M.D.  Ritchie,  B.C.  White,  J.S.  Parker  et  al.,  Optimizationof  neural  network  architecture  using  
genetic  programming  improvesdetection  and  modeling  of  gene-gene  interactions  in  studies  of  
humandiseases. BMC Bioinform. 4(1), 28 (2003). https:// doi. org/ 10. 1186/ 1471- 2105-4- 28 39. J. Russell, N. Pillay, A selection hyper-heuristic for transfer learning in genetic programming. Asso-
ciation  for  Computing  Machinery,  New  York,  NY,  USA,  GECCO  ’23  Companion,  pp.  631–634  
(2023). https:// doi. org/ 10. 1145/ 35831 33. 35906 86  40.  K.  Simonyan,  A.  Zisserman,  Very  deep  convolutional  networks  for  large-scale  image  recognition.  
1409.1556 (2015)  41.  N.  Srivastava,  G.  Hinton,  A.  Krizhevsky  et  al.,  Dropout:  a  simple  way  to  prevent  neural  networks  
from overfitting. J. Mach. Learn. Res. 15(1), 1929–1958 (2014)  42.  M.  Suganuma,  S.  Shirakawa,  T.  Nagao,  A  genetic  programming  approach  to  designing  convolu-
tional  neural  network  architectures.  In:  Proceedings  of  the  Twenty-Seventh  International  Joint  
