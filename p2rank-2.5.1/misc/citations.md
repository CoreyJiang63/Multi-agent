
# Publications

### 

If you use P2Rank, please cite relevant papers: 


* [Software article](https://doi.org/10.1186/s13321-018-0285-8) about P2Rank pocket prediction tool  
 Krivak R, Hoksza D. ***P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure.*** Journal of Cheminformatics. 2018 Aug.  
~~~bibtex
@article{p2rank,
    title={{P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure}},
    author={Kriv{\'a}k, Radoslav and Hoksza, David},
    journal={Journal of cheminformatics},
    volume={10},
    number={1},
    pages={39},
    year={2018},
    publisher={Nature Publishing Group},
    doi={10.1186/s13321-018-0285-8}
}
~~~

* [Latest web-server article](https://doi.org/10.1093/nar/gkaf421) about updates in P2Rank and [prankweb.cz](https://prankweb.cz)
 Polak L, Skoda P, Riedlova K, Krivak R, Novotny M, Hoksza D. ***PrankWeb 4: a modular web server for protein–ligand binding site prediction and downstream analysis.*** Nucleic Acids Research, 2025 May.
~~~bibtex
@article{prankweb4,
    author = {Polák, Lukáš and Škoda, Petr and Riedlová, Kamila and Krivák, Radoslav and Novotný, Marian and Hoksza, David},
    title = {PrankWeb 4: a modular web server for protein–ligand binding site prediction and downstream analysis},
    journal = {Nucleic Acids Research},
    pages = {gkaf421},
    year = {2025},
    month = {05},
    abstract = {Knowledge of protein–ligand binding sites (LBSs) is crucial for advancing our understanding of biology and developing practical applications in fields such as medicine or biotechnology. PrankWeb is a web server that allows users to predict LBSs from a given three-dimensional structure. It provides access to P2Rank, a state-of-the-art machine learning tool for binding site prediction. Here, we present a new version of PrankWeb enabling the development of both client- and server-side modules acting as postprocessing tasks on the predicted pockets. Furthermore, each module can be associated with a visualization module that acts on the results provided by both client- and server-side modules. This newly developed system was utilized to implement the ability to dock user-provided molecules into the predicted pockets using AutoDock Vina (server-side module) and to interactively visualize the predicted poses (visualization module). In addition to introducing a modular architecture, we revamped PrankWeb’s interface to better support the modules and enhance user interaction between the 1D and 3D viewers. We introduced a new, faster P2Rank backend or user-friendly exports, including ChimeraX visualization.},
    issn = {1362-4962},
    doi = {10.1093/nar/gkaf421},
    url = {https://doi.org/10.1093/nar/gkaf421},
    eprint = {https://academic.oup.com/nar/advance-article-pdf/doi/10.1093/nar/gkaf421/63227728/gkaf421.pdf},
}
~~~


* [Web-server article](https://doi.org/10.1093/nar/gkac389) about updates in the web interface [prankweb.cz](https://prankweb.cz)  
 Jakubec D, Skoda P, Krivak R, Novotny M, Hoksza D. ***PrankWeb 3: accelerated ligand-binding site predictions for experimental and modelled protein structures.*** Nucleic Acids Research, Volume 50, Issue W1, 5 July 2022, Pages W593–W597  
~~~bibtex
@article{prankweb3,
    title = "{PrankWeb 3: accelerated ligand-binding site predictions for experimental and modelled protein structures}",
    author = {Jakubec, David and Skoda, Petr and Krivak, Radoslav and Novotny, Marian and Hoksza, David},
    journal = {Nucleic Acids Research},
    volume = {50},
    number = {W1},
    pages = {W593-W597},
    year = {2022},
    month = {05},
    abstract = "{Knowledge of protein–ligand binding sites (LBSs) enables research ranging from protein function annotation to structure-based drug design. To this end, we have previously developed a stand-alone tool, P2Rank, and the web server PrankWeb (https://prankweb.cz/) for fast and accurate LBS prediction. Here, we present significant enhancements to PrankWeb. First, a new, more accurate evolutionary conservation estimation pipeline based on the UniRef50 sequence database and the HMMER3 package is introduced. Second, PrankWeb now allows users to enter UniProt ID to carry out LBS predictions in situations where no experimental structure is available by utilizing the AlphaFold model database. Additionally, a range of minor improvements has been implemented. These include the ability to deploy PrankWeb and P2Rank as Docker containers, support for the mmCIF file format, improved public REST API access, or the ability to batch download the LBS predictions for the whole PDB archive and parts of the AlphaFold database.}",
    issn = {0305-1048},
    doi = {10.1093/nar/gkac389},
}
~~~

* [Web-server article](https://doi.org/10.1093/nar/gkz424) introducing the web interface at [prankweb.cz](https://prankweb.cz)  
 Jendele L, Krivak R, Skoda P, Novotny M, Hoksza D. ***PrankWeb: a web server for ligand binding site prediction and visualization.*** Nucleic Acids Research, Volume 47, Issue W1, 02 July 2019, Pages W345-W349 
~~~bibtex
@article{prankweb,
    title="{{P}rank{W}eb: a web server for ligand binding site prediction and visualization}",
    author="Jendele, L.  and Krivak, R.  and Skoda, P.  and Novotny, M.  and Hoksza, D. ",
    journal="Nucleic Acids Res.",
    year="2019",
    volume="47",
    number="W1",
    pages="W345-W349",
    month="Jul",
    doi={10.1093/nar/gkz424}
}
~~~

* [Conference paper](https://doi.org/10.1007/978-3-319-21233-3_4) introducing P2Rank prediction algorithm  
 Krivak R, Hoksza D. ***P2RANK: Knowledge-Based Ligand Binding Site Prediction Using Aggregated Local Features.*** International Conference on Algorithms for Computational Biology 2015 Aug 4 (pp. 41-52). Springer
~~~bibtex
@inproceedings{p2rank-alcob,
    title={{P2RANK: Knowledge-Based Ligand Binding Site Prediction Using Aggregated Local Features}},
    author={Kriv{\'a}k, Radoslav and Hoksza, David},
    booktitle={International Conference on Algorithms for Computational Biology},
    pages={41--52},
    year={2015},
    organization={Springer},
    doi={10.1007/978-3-319-21233-3_4}
}
~~~

* [Research article](https://doi.org/10.1186/s13321-015-0059-5) about PRANK rescoring algorithm (now included in P2Rank)  
 Krivak R, Hoksza D. ***Improving protein-ligand binding site prediction accuracy by classification of inner pocket points using local features.*** Journal of Cheminformatics. 2015 Dec.
~~~bibtex
@article{prank,
    author={Kriv{\'a}k, Radoslav and Hoksza, David},
    title={Improving protein-ligand binding site prediction accuracy by classification of inner pocket points using local features},
    journal={Journal of Cheminformatics},
    year={2015},
    month={Apr},
    day={01},
    volume={7},
    number={1},
    pages={12},
    abstract={Protein-ligand binding site prediction from a 3D protein structure plays a pivotal role in rational drug design and can be helpful in drug side-effects prediction or elucidation of protein function. Embedded within the binding site detection problem is the problem of pocket ranking -- how to score and sort candidate pockets so that the best scored predictions correspond to true ligand binding sites. Although there exist multiple pocket detection algorithms, they mostly employ a fairly simple ranking function leading to sub-optimal prediction results.},
    issn={1758-2946},
    doi={10.1186/s13321-015-0059-5}
}
~~~
