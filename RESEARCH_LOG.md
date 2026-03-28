20240108<br>    finish the feature extraction code frame.

20240109<br>    finish the dataloader for new dataset, such as painting91, pandora etc.

20240110<br>    run the origianl dae project on painting91 for 1000 iterations<br>
                run one-time 1000 iterations on Painting91, the classification accuracy is only around 0.35<br>
                run one-time 1000 iterations on Painting91 adding classification loss, learning-rate=0.01, the results are not converged.

20240110<br>    run the original dae codes on painting91 with smaller learning rate, 0.00001 ... , see the CLS-accuracy  
                learning-rate=0.00001 CLS-accuracy=0.41  
                when lr=0.000001, 0.0000001, the CLS-accuracy is less than 0.30

20240110<br>    test multi linear layers for classification model  
                adding MLP model with lr=0.00001 CLS-accuracy=0.3664 two linear layers  
                adding MLP model with lr=0.00001 CLS-accuracy=0.2993 three linear layers  
                adding MLP model with lr=0.0001 CLS-accuracy=0.3523 three linear layers  
                adding MLP model with lr=0.0001 CLS-accuracy=0.3422 single linear layers, batch=384   
                adding MLP model with lr=0.00001 CLS-accuracy=0.2596 four linear layers, batch=384  
                adding MLP model with lr=0.0001 CLS-accuracy=0.3191 four linear layers, batch=384  
                adding MLP model with lr=0.001 CLS-accuracy=0.2890 four linear layers, batch=384  
                adding MLP model with lr=0.00001 CLS-accuracy=0.3366 two linear layers, batch=384  
                lr=0.0001 and two layers are the current best option

20240112<br>    delete the noise adding function on (1)<br>
                adding MLP model with lr=0.00001 CLS-accuracy=0.3593 two linear layers, batch=384  
                adding MLP model with lr=0.00001 CLS-accuracy=0.4085 two linear layers, batch=256  

20240112<br>    add the cdae module into 5, test the accuracy  
                strider = 2, adding to cdae=5 lr=0.00001 CLS-accuracy=0.3122 two linear layers, batch=256  
                when the layer number is increased, the converge rate is slower, maybe 2000 or more will be better than 1000  
                strider = 2, kernel_size = 4, padding = 1, adding to cdae=5 lr=0.00001 CLS-accuracy=0.3132 two linear layers, batch=256  
                strider = 2, kernel_size = 4, padding = 1, lr = 0.05 adding to cdae=5 lr=0.00001 CLS-accuracy=0.2832 two linear layers, batch=256  
                only increasing cdae number maybe is not a good idea.

20240113<br>    add more Conv2d layer for single cdae into 2, test the accuracy   
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.0001 CLS-accuracy=0.3894 two linear layers, batch=256  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 CLS-accuracy=0.1621 two linear layers, batch=256, no nomalization  
                when the normalization is removed, the reconstructed images are not converged

20240113<br>    change the input into feature mode  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 CLS-accuracy=0.3808 two linear layers, batch=256, no nomalization  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 CLS-accuracy=0.4644 two linear layers, batch=256, no nomalization CLS-backwards=20  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 CLS-accuracy=0.5023 three linear layers, batch=256, no nomalization CLS-backwards=30  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 CLS-accuracy=0.6387 three linear layers, batch=256, no nomalization CLS-backwards=50, 5000

20240113<br>    test different loss function in sae model.py  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization CLS-backwards=100, 1000  
                SmoothL1Loss CLS-accuracy=0.6390<br>
                MSELoss CLS-accuracy=0.6320<br>
                HuberLoss CLS-accuracy=0.6156<br>

20240113<br>    test Resnet50 feature from SJC, failed. the reason is the features are not compatible.  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization CLS-backwards=30, 1000

20240118<br>    add setLevels function, the default is 'Random'  
                
20240118<br>    add level attribute in Dataset

20240118<br>    make the generated new features into the circle

20240118<br>    add setLevels function in SCLdataset.py

20240118<br>    combine vgg16 and resnet50 sae class into one unified class  
                combine image and features sae class into one unified class

20240118<br>    change the specified initialization process into the auto initialization with legal variables  
                The results are very promising  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.000001 three linear layers, batch=256, no nomalization CLS-backwards=10, 1000, CLS-accuracy=0.2507  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.000001 three linear layers, batch=256, no nomalization CLS-backwards=30, 1000, CLS-accuracy=0.3226  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization CLS-backwards=50, 1000, CLS-accuracy=0.5257  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization CLS-backwards=100, 5000, CLS-accuracy=0.6300  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization one ae CLS-backwards=100, 1000, CLS-accuracy=0.9132  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization two ae CLS-backwards=100, 1000, CLS-accuracy=0.7792  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 three linear layers, batch=256, no nomalization three ae CLS-backwards=100, 1000, CLS-accuracy=0.5964  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 ae_lr=0.01 three linear layers, batch=256, no nomalization three ae CLS-backwards=100, 1000, CLS-accuracy=0.6103

20240119<br>    add ae2 ae3

20240119<br>    re-write CDAutoEncoder class, the backward_pass should be changed
                level1 = festures, level2 = level1, level3 = level2  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 ae_lr=0.01 three linear layers, batch=256, no nomalization three ae CLS-backwards=100, 1000, CLS-accuracy=0.6289  
                ae_lr=0.01 maybe fall in the local optimal easily. 0.1 is the best option for now  
                more than 5000 iterations test has been done, there is no good news. The reason is probably the style level setting method.  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 ae_lr=0.01 three linear layers, batch=256, no nomalization one ae CLS-backwards=100, 3000, CLS-accuracy=0.8792  
                level1 = features, level2 = features, level3 = features  
                strider = 2, kernel_size = 2, padding = 0, adding to cdae=3 lr=0.00001 ae_lr=0.01 three linear layers, batch=256, no nomalization one ae CLS-backwards=100, 3000, CLS-accuracy=0.8960  
                The reason of lower accuracy is probably the style level setting method. 

20240120<br>    compute the average centroid

20240120<br>    compute the distance from centroid

20240120<br>    finish the style level set based on Euclidean distance  
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=0.1 three layers, batch=256, no nomalization one ae CLS=100, 1000, Euclidean, CLS-accuracy=0.9128  
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=0.01 three layers, batch=256, no nomalization three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.6113
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=variable three layers, batch=256, no nomalization three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.6574  
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=smaller variable three layers, batch=256, nomalization three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.7656
                set the classifier do not initialization during the training process, the accuracy will be increased obviously.

20240120<br>    add classification loss into the sae loss, see what will happen  
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=smaller variable three layers, batch=256, nomalization CLS+RECON three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.7660  
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=0.1 three layers, batch=256, nomalization CLS+RECON three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.6898
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=variable three layers, batch=256, nomalization CLS+RECON three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.6906
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=smaller variable three layers, batch=256 CLS+RECON nomalization three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.8765

20240121<br>    customize loss class in 'utils/'<br>

20240121<br>    try KL-diverge algorithm to calculate the style similarity<br>
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=0.0001 three layers, batch=256, nomalization CLS+RECON+KL three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.7472<br>
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=0.00001 three layers, batch=256, nomalization CLS+RECON+KL three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.8102<br>
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=0.000001 three layers, batch=256, nomalization CLS+RECON+KL three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.7836<br>
                The KL-diverge make the max and mean is smaller, and the sparsity is bigger<br>

20240121<br>    try cos_similarity to calculate the style similarity
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=original three layers, batch=256, nomalization CLS+RECON+CS three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.6214<br>
                0.7664 <br>

20240121<br>    try cityblock distance<br>
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=original three layers, batch=256, nomalization CLS+RECON+chebyshev three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.7828<br>
                0.7832 <br>

20240121<br>    try DotProductSimilarity<br>
                strider=2, kernel_size=2, padding=0, adding to cdae=3 lr=0.00001 ae_lr=original three layers, batch=256, nomalization CLS+RECON+XX three ae CLS=100, 1000, Euclidean, CLS-accuracy=0.7055<br>

20240121<br>    try the diamond type of SLAE add diamond sae<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.00001 batch=256, nomalization RECON one dsc CLS=always, 1000, Euclidean, CLS-accuracy=0.9133<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.1 batch=256, nomalization RECON two dsc CLS=always, 1000, Euclidean, CLS-accuracy=0.4976<br>
                obviously, the speed of converge is slower, more iterations are required.<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.1 batch=256, nomalization RECON two dsc CLS=always, 3000, Euclidean, CLS-accuracy=0.7625<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON two dsc CLS=always, 5000, Euclidean, LeakReLU CLS-accuracy=0.6869<br>

20240121<br>    try ELU<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON two dsc CLS=always, 2000, Euclidean, ELU CLS-accuracy=0.7292<br>
                The iteration is 2000 maybe the best option

20240122<br>    try SiLU<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON two dsc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7879<br>
                SiLU is the best option for now<br>

20240122<br>    try Mish
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON two dsc CLS=always, 2000, Euclidean, Mish CLS-accuracy=0.6246<br>
                Mish, the Max is increased extremely.<br>

20240122<br>    add dsc1, dsc2, dsc3<br>
                strider=2, kernel_size=2, padding=0, adding to dsc=1 lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON three dsc CLS=always, 5000, Euclidean, SiLU CLS-accuracy=0.5710<br>
                when the layers are increased, the lr and iterations should be increased as well.<br>

20240123<br>    try stacked dsc1<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON three dsc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.6133<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON+CLS three dsc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.5199<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=smaller variable batch=256, nomalization RECON+CLS+CS three dsc CLS=always, 2000, Euclidean, SiLU CLS-accuracy FAILED<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON+CLS+KL three dsc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=FAILED<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization RECON+CLS+SmoothL1Loss three dsc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=FAILED<br>
                only MSEloss is workable. Others are failed all, when the layers are three and deeper.<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=original batch=256, nomalization SmoothL1Loss+CLS dsc1 CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.9129<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=smaller batch=256, nomalization SmoothL1Loss+CLS+KL dsc1 CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.8738<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization SmoothL1Loss+CLS+Cosin dsc1 CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.9133<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization SmoothL1Loss+CLS+Cosin dsc1 CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.9129<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.0001 batch=256, nomalization SmoothL1Loss+CLS+Cosin dsc1-dsc1 CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8133<br>
                still not converge, bigger ae_lr is better
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dsc1-dsc1 CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7375<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dsc1-dsc1 CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.8047<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+cityblock dsc1-dsc1 CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7140<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dsc1-dsc1[2] CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7656<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dsc1-dsc1[2] CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.8051<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization (CLS+Cosin)/2 dsc1-dsc1[2] CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7507<br>

20240123<br>    re-write dsc1, l4->l3, l3->l2, l2->l1, l1->l1, re-do the upper two steps, see the CLS-accuracy<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.9129<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7023<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7308<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc[2] CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7089<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc[2] getLeveNew CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.6203<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2] getLeveNew2 retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.6770<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2] getLeveNew2 retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7691<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2] getLeveNew2 retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7300<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc getLeveNew2 retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.9133<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.9129<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.6312<br>
                
20240124<br>    strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 5000, Euclidean, SiLU CLS-accuracy=0.8394<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.005 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 9000, Euclidean, SiLU CLS-accuracy=0.6918<br>                         
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew2 retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.7066<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew2 retain_graph CLS=always, 7000, Euclidean, SiLU CLS-accuracy=0.7265<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 5000, Euclidean, SiLU CLS-accuracy=0.7453 <br>

20240124<br>    try the parallel dsc model under the current circumstance
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.8793<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.0001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8387<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.9133<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8613<br>

20240124<br>    try the jump-connection between different dsc cells
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8160<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 2000, Euclidean, SiLU CLS-accuracy=0.8613<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7847<br>
                jump every dslc
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7012<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6750<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7336<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.0001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7835<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.9133<br>
                for the jump dslc, the smaller learning rate , the better results, I guess the best performance is 0.9133, check the wrong image later for analysis

20240125<br>    try painting91 resnet50 2048x7x7, see the accuracy 
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6922<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.000001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6188<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6922<br>
                jump-connection one-by-one
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.9129<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7656<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc-dslc[2]-dslc[3] getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8793<br>

20240125<br>    try pandora on vgg16 features
                mode 1 single dslc
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8567<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8543<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.0001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8515<br>
                mode 2 dslc-dslc-dslc
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.0001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.4808<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.<br>
                mode 3 dslc-dslc[2]-dslc[3]
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.4917<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6003<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.4984<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 7000, Euclidean, SiLU CLS-accuracy=0.7766<br>
                mode 4 common jump connection
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8512<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8543<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8009<br>
                model 5 jump-connection one-by-one
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8308<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6906<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Cosin dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8615<br>
                
20240126<br>    add gram distance inside the dsc class
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.6614<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7669<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8458<br>
                mode 1 single dslc
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8575<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8524<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8524<br>
                no CLS
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8552<br>
                with layerNorm see the accuracy
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.01 batch=256, nomalization Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.8560<br>
                strider=2, kernel_size=2, padding=0, lr=0.00001 ae_lr=0.0001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU CLS-accuracy=0.7542<br>
                strider=2, kernel_size=4, padding=0, lr=0.00001 ae_lr=0.00001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 500, Euclidean, SiLU layerNormal CLS-accuracy=0.8548<br>
                new loss : loss1 and loss2
                strider=2, kernel_size=4, padding=0, lr=0.00001 ae_lr=0.001 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 500, Euclidean, SiLU layerNormal CLS-accuracy=0.8398<br>
                strider=2, kernel_size=4, padding=0, lr=0.00001 ae_lr=0.1 batch=256, nomalization CLS+Gram dslc getLeveNew retain_graph CLS=always, 1000, Euclidean, SiLU layerNormal CLS-accuracy=0.8398<br>


Arch
CLS_accuracy = 0.9535<br>

FashionStyle14
CLS_accuracy = 0.9218<br>

WikiArt3
CLS_accuracy = 0.9429<br>

202420207<br>   try to split train and test set see the accuracy in different dataset<br>
                AVAstyle-training 0.9797<br>
    
