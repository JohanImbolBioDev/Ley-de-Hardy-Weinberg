# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/14HUz5WSOBlpghSb9v-G_40cYTER5XRWF 
"""

class Alelos:
    def __init__(self, A, B, O, AB):
        self.A = A
        self.B = B
        self.O = O
        self.AB = AB

    @property
    def total(self):
        return self.A + self.B + self.O + self.AB
#Frecuencias relativas
    @property
    def fA(self):
        return self.A / self.total

    @property
    def fB(self):
        return self.B / self.total

    @property
    def fO(self):
        return self.O / self.total

    @property
    def fAB(self):
        return self.AB / self.total

import math
import numpy as np
from scipy.stats import chi2
import pandas as pd
from scipy.stats import chisquare

class LeyHardyWeinberg:
    def __init__(self, alelos, C_Yates=False, C_Bernstein = False):
        self.alelos = alelos
        self.C_Yates = C_Yates  #Aplicar Corrección de Yates (Muestras pequeñas)
        self.C_Bernstein = C_Bernstein  #Aplicar Corrección de Bernstein
        self.AB_esperado = self.frecuencias_fenotipicas_absolutas_esperadas()['AB']

    @property
    def get_Alelos(self):
        return {
            'A' : self.alelos.A,
            'B' : self.alelos.B,
            'O' : self.alelos.O,
            'AB' : self.alelos.AB,
            'Total' : self.alelos.total,
        }

    @property
    def r(self):
        return math.sqrt(self.alelos.fO)

    @property
    def p(self):
        return 1 - math.sqrt(self.alelos.fB + self.alelos.fO)

    @property
    def q(self):
        return 1 - math.sqrt(self.alelos.fA + self.alelos.fO)

    @property
    def D_correcion(self):
        return 1 - (self.p + self.q + self.r)

    @property
    def Abchi(self):
      return ((self.alelos.AB  - self.AB_esperado)**2 ) / self.AB_esperado

    def Correccion_bernstein(self, _p, _q, _r ,_D):
        p = _p * (1 + (_D/2))
        q = _q * (1 + (_D/2))
        r = (_r + (_D/2)) * (1 + (_D/2))

        return p, q, r

    def Correcion_Yates(self, X, Xesperado):
        return (((X  - Xesperado)-0.5)**2 ) / Xesperado


    def frecuencias_fenotipicas_absolutas_esperadas(self):
        p, q, r, D = self.p, self.q, self.r, self.D_correcion

        if self.C_Bernstein is True:
            p, q, r = self.Correccion_bernstein(p, q, r, D)

        A = (p ** 2) + (2 * p * r)
        B = (q ** 2) + (2 * q * r)
        O = r **2
        AB = 2 * p * q

        expected_frequencies = {
            'A' : self.alelos.total * A,
            'B' : self.alelos.total * B,
            'O' : self.alelos.total * O,
            'AB' : self.alelos.total * AB
        }

        total_expected = sum(expected_frequencies.values())
        total_observed = self.alelos.total

        if not math.isclose(total_expected, total_observed, rel_tol=1e-8):
            adjustment_factor = total_observed / total_expected
            expected_frequencies = {key: value * adjustment_factor for key, value in expected_frequencies.items()}

        return expected_frequencies

    def getHardyWeinberg(self):
        X2 = self.Abchi
        if self.C_Yates is True:
          X2 = self.Correcion_Yates(self.alelos.AB, self.AB_esperado)

        p_value =  round(chi2.sf(X2, 1),3)

        return {
            'Chi-Square' : round(X2,3),
            'p-value' : round(p_value,3),
            'gl': 1
        }

    def Chi_Combinada(self):
        Aexp, Bexp, Oexp, ABexp = self.frecuencias_fenotipicas_absolutas_esperadas().values()
        stat, p = chisquare([self.alelos.A, self.alelos.B, self.alelos.O, self.alelos.AB], [Aexp, Bexp, Oexp, ABexp])
        return {
            'Chi-Square' : stat,
            'p-value' : p
        }

class ResultadosHardyWeinberg:
  def __init__(self, *alelos, C_Bernstein = False, C_Yates = False):
    self.alelos = alelos
    self.C_Bernstein = C_Bernstein
    self.C_Yates = C_Yates
    self.L_hardy = [LeyHardyWeinberg(i, C_Yates= self.C_Yates, C_Bernstein=False) for i in self.alelos[0]]
    self.resultados = pd.DataFrame({
        'N°':[] ,'A' : [], 'B' : [], 'O' : [], 'AB' : [], 'Total' : [], 'Chi-Square' : [],
        'p-value' : [], 'gl': []
    })

    self.Calcular_resultados()


  def Calcular_resultados(self):
    for i in self.L_hardy:
      resultado_chi = i.getHardyWeinberg()
      self.resultados.loc[len(self.resultados)] = [
          len(self.resultados) + 1, i.get_Alelos['A'], i.get_Alelos['B'], i.get_Alelos['O'],
          i.get_Alelos['AB'], i.get_Alelos['Total'], round(resultado_chi['Chi-Square'],3),
          round(resultado_chi['p-value'],3), int(resultado_chi['gl'])
      ]

class heterogeneidad:
  def __init__(self, *alelos, C_Bernstein = False, C_Yates = False):
    self.C_Bernstein = C_Bernstein
    self.C_Yates = C_Yates
    self.alelos = alelos
    self.L_hardy = [LeyHardyWeinberg(i, C_Bernstein=self.C_Bernstein, C_Yates=self.C_Yates) for i in self.alelos[0]] # Access the list of Alelos objects

    self.resultadosChi = pd.DataFrame({
        'N°':[] ,'A' : [], 'B' : [], 'O' : [], 'AB' : [], 'Total' : [], 'Chi-Square' : [],
        'p-value' : [], 'gl': []
    })
    self.Calcular_resultadosChi()
    self.SumaChi = self.Calcular_SumaChi()
    self.SumaGl = self.Calcular_SumaGl()

    self.AlelosCombinados = Alelos(self.resultadosChi['A'].sum(),
                                   self.resultadosChi['B'].sum(),
                                   self.resultadosChi['O'].sum(),
                                   self.resultadosChi['AB'].sum())

    self.LeyCombinada = LeyHardyWeinberg(self.AlelosCombinados, C_Bernstein=self.C_Bernstein, C_Yates=self.C_Yates)

    self.resultadosCombinada = pd.DataFrame({
        'N°':[] ,'A' : [], 'B' : [], 'O' : [], 'AB' : [], 'Total' : [], 'Chi-Square' : [],
        'p-value' : [], 'gl': []
    })

    self.Calcular_CombinadaChi()

    self.Diferencia_Heterogeneidad = self.CalcularDiferenciaHeterogeneidad()
    self.Diferencia_GL = self.CalcularDiferenciaGL()
    self.resultados = self.get_resultados()

  def Calcular_resultadosChi(self):
    for i in self.L_hardy:
      resultado_chiCombinada = i.Chi_Combinada()
      self.resultadosChi.loc[len(self.resultadosChi)] = [
          len(self.resultadosChi) + 1,
          i.get_Alelos['A'],
          i.get_Alelos['B'],
          i.get_Alelos['O'],
          i.get_Alelos['AB'],
          i.get_Alelos['Total'],
          round(resultado_chiCombinada['Chi-Square'],3),
          round(resultado_chiCombinada['p-value'],3),
          int(3)
      ]

  def Calcular_CombinadaChi(self):
    resultado_chiCombinada = self.LeyCombinada.Chi_Combinada()
    self.resultadosCombinada.loc[len(self.resultadosCombinada)] = [
        'X2 Combinado',
        self.LeyCombinada.get_Alelos['A'],
        self.LeyCombinada.get_Alelos['B'],
        self.LeyCombinada.get_Alelos['O'],
        self.LeyCombinada.get_Alelos['AB'],
        self.LeyCombinada.get_Alelos['Total'],
        round(resultado_chiCombinada['Chi-Square'],3),
        '',
        int(1)
    ]


  def Calcular_SumaChi(self):
    return self.resultadosChi['Chi-Square'].sum()

  def Calcular_SumaGl(self):
    return self.resultadosChi['gl'].sum()

  def CalcularDiferenciaHeterogeneidad(self):
    return self.SumaChi - self.resultadosCombinada['Chi-Square'][0]

  def CalcularDiferenciaGL(self):
    return self.SumaGl - self.resultadosCombinada['gl'][0]

  def get_resultados(self):
      resultados_combinados = pd.concat([self.resultadosChi, self.resultadosCombinada], ignore_index=True)
      resultados_combinados.loc[len(resultados_combinados)] = ['Suma X2', '', '', '', '', '', self.SumaChi, '', self.SumaGl]
      resultados_combinados.loc[len(resultados_combinados)] = ['Diferencia Heterogeneidad', '', '', '', '', '',
                                                               self.Diferencia_Heterogeneidad, '', self.Diferencia_GL]
      return resultados_combinados
