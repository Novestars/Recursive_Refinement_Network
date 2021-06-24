# RRN:Recursive Refinement Network: learning based deformable image registration

By Xinzi He, Jia Guo, Xuzhe Zhang, Hanwen Bi, Sarah Gerard, David Kaczka, Amin Motahari, Eric Hoffman, Joseph Reinhardt, R. Graham Barr, Elsa Angelini, and Andrew Laine.

Paper link: [arXiv] https://arxiv.org/pdf/2106.07608.

Unsupervised learning-based medical image registration ap- proaches have witnessed rapid development in recent years. We propose to revisit a commonly ignored while simple and well-established princi- ple: recursive refinement of deformation vector fields across scales. We introduce a recursive refinement network (RRN) for unsupervised med- ical image registration, to extract multi-scale features, construct nor- malized local cost correlation volume and recursively refine volumetric deformation vector fields. RRN achieves state of the art performance for 3D registration of expiratory-inspiratory pairs of CT lung scans. On DirLab COPDGene dataset, RRN returns an average Target Reg- istration Error (TRE) of 0.83 mm, which corresponds to a 13% error reduction from the best result presented in the leaderboard 4. In addi- tion to comparison with conventional methods, RRN leads to 89% error reduction compared to deep-learning-based peer approaches.

Recurive Refinement Network is constructed for image alignment and registartion. 
Thanks to previous open-sourced repo:  
[VoxelMorph](https://github.com/voxelmorph/voxelmorph)    
[UFlow](https://github.com/google-research/google-research/tree/master/uflow)   
