agndiag
=======

![CI](https://github.com/nudomarinero/agndiag/actions/workflows/main.yml/badge.svg)

Software to apply the AGN diagnostic diagrams to emission line data. 

Currently there are three diagnostic diagrams implemented from the paper of Sabater et al. 2012:
* diag_nii_Sabater2012 - [NII] diagnostic diagram
* diag_sii_Sabater2012 - [SII] diagnostic diagram
* diag_oi_Sabater2012  - [OI] diagnostic diagram
If the keyword use_limits is set to true, the censored data is taken into account as in the original paper.

There are some functions to obtain a final classification from the classification given by the individual diagnostic diagrams:
* diag_class_Sabater2012  - Criteria of Sabater et al. 2012 
* diag_class_OiSiiNiiMine - New improved criteria

Finally, the classification criteria of Cid-Fernandes et al. 2011 was implemented:
* diag_CidFernandes2011