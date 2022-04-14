<div id="top"></div>


<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">OCEL Generic Feature Extraction</h3>

  <p align="center">
    Various tools and structures in order to extract the most explicit and implicit data from the OCEL event log format.
  </p>
</div>



<!-- TABLE OF CONTENTS -->
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#installation-and-usage">Installation and Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>




<!-- ABOUT THE PROJECT -->
## About The Project
This repository belongs to a master thesis project titled ["A generic approach towards feature extraction from object-centric event logs"](https://www.pads.rwth-aachen.de/cms/PADS/Studium/Abschlussarbeiten/~ppsxz/Master-Thesis-A-generic-approach-towar/lidx/1/) hosted by the PADS group at RWTH Aachen. Various methods are used in order to make extracting information from object-centric event logs simple and useful. Anyone with a valid object-centric event log should be able to use all the functions and methods defined in the repository to gather more information about their processes. Please check out some examples and if you have any questions or additional ideas, feel free to send me a message.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Python 3.6+](https://www.python.org/)
* [ocel-support](http://ocel-standard.org/)
* [networkx](https://networkx.org/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [flake8](https://flake8.pycqa.org/en/latest/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
### Installation and Usage
As this is a master thesis project, not too much effort is being put into streamlining the installation process for the code. This will perhaps be done if there is interest/I have the time after the actual graded thesis is completed. For some manual installation instructions, please see below:

1. Clone the repo
   ```sh
   git clone https://github.com/DerAndereJohannes/ocel-features
   ```
2. Create a new venv and install any requirements
   ```sh
   pip install -r requirements.txt
   ```
3. Add the `ocel-features` package into your PYTHONPATH environment variable (Optional: add this to your .bashrc file)
   ```sh
   export PYTHONPATH="/path/to/ocel-features:"$PYTHONPATH
   ```
4. Finished! Check out some examples and enjoy! :-)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap
The overall roadmap is far too large to be written down in this section. I have listed below the parts that I am actively working on in this repository or am planning to tackle in the near future.

- [ ] Object Features
  - [ ] Point
  - [ ] Global/Group
- [ ] Event Features
  - [ ] Point
  - [ ] Global/Group
- [ ] Situation Features
  - [ ] Based on different time perspectives
  - [ ] Based on different relations
- [ ] OCEL Structures
  - [ ] Object-Object Graphs
  - [ ] OCEL Graphs
  - [ ] Event Graphs
- [ ] Feature Series
  - [ ] Based on Log
  - [ ] Based on Graphs
- [ ] OCEL Decomposition for Local Feature Extractions
  - [ ] Decompose Log
  - [ ] Decompose General OCEL graphs
  - [ ] Decompose Time Concious Graphs


See the [open issues](https://github.com/DerAndereJohannes/ocel-features/issues) for a full list of proposed features (and known issues). Feel free to also leave some extra ideas too!

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Johannes Herforth - [Website/Contact](https://node.lu/)

Project Link: [https://github.com/DerAndereJohannes/ocel-features](https://github.com/DerAndereJohannes/ocel-features)

<p align="right">(<a href="#top">back to top</a>)</p>

