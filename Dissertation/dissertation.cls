\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dissertation}[2012/05/04, Daniel Page <Daniel.Page@bristol.ac.uk>]

% =============================================================================

% parse class options

\RequirePackage{pgfkeys}
\RequirePackage{pgfopts}


\pgfkeys{/dissertation/.cd,
%     author = string
      author/.store in = \dissertation@author,
      author           = {},
% supervisor = string
  supervisor/.store in = \dissertation@supervisor, 
  supervisor           = {},
%     degree = [BSc|MEng|MSci|MSc|PhD]
      degree/.store in = \dissertation@degree,
      degree           = {MEng},
%      title = string
       title/.store in = \dissertation@title@major,
       title           = {},
%   subtitle = string
    subtitle/.store in = \dissertation@title@minor,
    subtitle           = {},
%       type = string
        type/.store in = \dissertation@type,
        type           = {},
%       type = string
        year/.store in = \dissertation@year,
        year           = {\the\year}
}

\ProcessPgfOptions{/dissertation}

% load base class, then general-purpose packages

\LoadClass[10pt,a4paper,twoside]{book}

% http://www.ctan.org/tex-archive/macros/latex/required/graphics
\RequirePackage{graphicx}
% http://www.ctan.org/tex-archive/macros/latex/required/amslatex
\RequirePackage{amsmath}
\RequirePackage{amssymb}
% http://www.ctan.org/tex-archive/macros/latex/contrib/adjustbox
\RequirePackage{adjustbox}
% http://www.ctan.org/tex-archive/macros/latex/contrib/etoolbox
\RequirePackage{etoolbox}
% http://www.ctan.org/tex-archive/macros/latex/contrib/fancyhdr
\RequirePackage{fancyhdr}
% http://www.ctan.org/tex-archive/macros/latex/contrib/geometry
\RequirePackage{geometry}
% http://www.ctan.org/tex-archive/macros/latex/contrib/datetime
\RequirePackage{datetime}
% http://www.ctan.org/tex-archive/macros/latex/contrib/algorithm2e
\RequirePackage[algochapter]{algorithm2e}
% http://www.ctan.org/tex-archive/macros/latex/contrib/listings
\RequirePackage{listings}
% http://www.ctan.org/tex-archive/macros/latex/contrib/xstring
\RequirePackage{xstring}
% http://www.ctan.org/tex-archive/macros/latex/contrib/url
\RequirePackage{url}
% http://www.ctan.org/tex-archive/macros/latex/contrib/hyperref
\RequirePackage{hyperref}
% http://www.ctan.org/tex-archive/macros/latex/contrib/lipsum
\RequirePackage{lipsum}
% http://www.ctan.org/tex-archive/usergrps/dante/dtk
\RequirePackage{dtklogos}
\RequirePackage{float}
\RequirePackage[parfill]{parskip}

% =============================================================================

\makeatletter

% -----------------------------------------------------------------------------

% configure hyperref

\hypersetup{colorlinks=true,bookmarks=false,urlcolor=red}

% configure algorithm2e

\DontPrintSemicolon

% configure listings

\lstset{basicstyle=\ttfamily,captionpos=b}
\renewcommand{\lstlistlistingname}{List of Listings}

% -----------------------------------------------------------------------------

% configure style of table of contents

\setcounter{tocdepth}{1}

% configure style of bibliography

\bibliographystyle{plain}

% configure style of pages

\geometry{a4paper,lmargin=2.5cm,rmargin=2.5cm,tmargin=2.5cm,bmargin=2.5cm}

\fancypagestyle{plain}{
  \fancyhf{}

  \renewcommand{\headrulewidth}{0.5pt}
  \renewcommand{\footrulewidth}{0.5pt}

  \fancyfoot[C]{\thepage}
}

\fancypagestyle{marked}{
  \fancyhf{}

  \renewcommand{\headrulewidth}{0.5pt}
  \renewcommand{\footrulewidth}{0.5pt}

  \fancyhead[LO]{\slshape \rightmark}
  \fancyhead[RE]{\slshape  \leftmark}

  \fancyfoot[C]{\thepage}
}

\pagestyle{plain}

% -----------------------------------------------------------------------------

% define macro to generate UoB title page

\renewcommand{\maketitle}{
  \newpage
  \thispagestyle{empty}

  \IfEq{\dissertation@type}{}{}{
    \adjustbox{padding={5pt},frame={1pt},right}{Dissertation Type: {\dissertation@type}}
  }

  \vspace*{\fill}
  \begin{center}
                \includegraphics[scale=0.3]{logo/logo_uob_color}                \\
                              \vspace*{1.0cm}
                          DEPARTMENT OF COMPUTER SCIENCE                        \\
                              \vspace*{2.0cm}
  \mbox{\vbox{\IfEq{\dissertation@title@major}{}{}{\LARGE\protect\dissertation@title@major}}} \\
                              \vspace*{0.5cm}
  \mbox{\IfEq{\dissertation@title@minor}{}{}{\Large \dissertation@title@minor}} \\
                              \vspace*{1.0cm}
                          {\Large \dissertation@author}                         \\

                              \vspace*{1.0cm}
                          \rule{0.5\textwidth}{0.5pt}
                              \vspace*{1.0cm}

            A dissertation submitted to the University of Bristol
            in accordance with the requirements of the degree of
                    \IfEqCase*{\dissertation@degree}{
                      {BSc} {Bachelor of     Science}
                      {MEng}{Master   of Engineering}
                      {MSci}{Master   of     Science}
                      {MSc} {Master   of     Science}
                      {PhD} {Doctor   of Philosophy}
                    }~in the Faculty of Engineering.                                

                              \vspace*{1.0cm}
                          \rule{0.5\textwidth}{0.5pt}
                              \vspace*{1.0cm}

                                  \today
  \end{center}
  \vspace*{\fill}
}

% define macro to generate UoB declaration

\newcommand{\makedecl}{
  \newpage
  \thispagestyle{plain}

  \chapter*{Declaration}

  This dissertation is submitted to the University of Bristol in accordance 
  with the requirements of the degree of {\dissertation@degree} in the Faculty 
  of Engineering.  It has not been submitted for any other degree or diploma 
  of any examining body.  Except where specifically acknowledged, it is all 
  the work of the Author. 
  \includegraphics[]{signature.png}

  \noindent {\dissertation@author}, \today
}

% define macros to mark front, main and back matter

\renewcommand{\frontmatter}{
  \cleardoublepage
  \pagestyle{plain}
  \pagenumbering{roman}
}
\renewcommand{\mainmatter}{
  \cleardoublepage
  \pagestyle{marked}
  \pagenumbering{arabic}
}
\renewcommand{\backmatter}{
  \cleardoublepage
  \pagestyle{marked}
}

% -----------------------------------------------------------------------------

\makeatother

% =============================================================================


