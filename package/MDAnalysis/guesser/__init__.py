# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the Lesser GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#

"""
Context-specific guessers --- :mod:`MDAnalysis.guesser`
========================================================

You can use guesser classes either directly by initiating an instance of it and use its guessing methods or through
the :meth:`guess_TopologyAttrs <guess_TopologyAttrs>`: API of the universe.

The following table lists the currently supported Context-aware Guessers along with
the attributes they can guess.

.. table:: Table of Supported Guessers

   ============================================== ========== ===================== ===================================================
   Name                                           Context    Attributes            Remarks
   ============================================== ========== ===================== ===================================================
   :ref:`DefaultGuesser <DefaultGuesser>`         default    types, elements,      general purpose guesser                                                                                       
                                                             masses, bonds,
                                                             angles, dihedrals,
                                                             improper dihedrals
   ============================================== ========== ===================== ===================================================
                        
                                


"""
from . import base
from .default_guesser import DefaultGuesser
