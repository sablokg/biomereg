# biomereg

- regression modelling for the microbiome datasets
- Centered log-ratio transform -> pseudocounts -> permutation -> regression modelling.

### Entire regression modelling in Rust for the microbiome. 


```
cargo build
```


```

                                                                                                                      
                               _|         _|                                          _|_|_|     _|_|_|_|     _|_|_|  
                               _|_|_|            _|_|     _|_|_|  _|_|       _|_|     _|    _|   _|         _|        
                               _|    _|   _|   _|    _|   _|    _|    _|   _|_|_|_|   _|_|_|     _|_|_|     _|  _|_|  
                               _|    _|   _|   _|    _|   _|    _|    _|   _|         _|    _|   _|         _|    _|  
                               _|_|_|     _|     _|_|     _|    _|    _|     _|_|_|   _|    _|   _|_|_|_|     _|_|_|  
                                                                                                                      
                                                                                                                      
Regression Modelling for microbiome
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************

Usage: biomereg <COMMAND>

Commands:
  random         Random modelling regression
  logistic       Logistic modelling regression
  knn            KNN modelling regression
  xg-boost       XGB modelling regression
  tsne           tsne modelling regression
  decision-tree  Decision modelling regression
  help           Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```
