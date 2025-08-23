#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Script: alface_roxa.py
# Created By: Amanda Lais de Freitas
# Created Date: 2025-08-21
# Version: 1.0
# ---------------------------------------------------------------------------
"""
Este módulo é responsável por [...]. 
Ele realiza as seguintes operações: [...].
"""
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import sys



### obrigado por fazer o exercício sobre cabeçalhos, espero que tenha sido ilustrativo... 
### Qual a diferença de cabeçalho e DocString no python?

###  abaixo segue desafio... 

## dica: pesquisa operacional...

import numpy as np
from itertools import combinations

EPS = 1e-9

#  1/3) Modele A, b, c 
# Forma padrão: A @ [x,y] <= b, com não-negatividade via -I @ [x,y] <= 0
# Restrição 1: 2x +  y <= 100
# Restrição 2:  x + 2y <= 80
# Restrição 3:  x >= 0  ->  -x <= 0
# Restrição 4:  y >= 0  ->  -y <= 0
A = np.array([
    # preencha aqui
], dtype=float)

b = np.array([
    # preencha aqui
], dtype=float)

# Função objetivo: z = 3x + 5y
c = np.array([
    # preencha aqui
], dtype=float)


#  2/3) Interseção de duas retas
# Cada reta é dada por (a, beta) representando a @ [x,y] = beta
def intersecao(reta1, reta2):
    a1, b1 = reta1
    a2, b2 = reta2
    Aeq = np.vstack([a1, a2])        # 2x2
    beq = np.array([b1, b2], float)  # 2,
    # Se forem paralelas/singulares, retorne None
    # Dica: use det ou tente resolver e capture exceção
    # TODO:
    raise NotImplementedError


#  3/3) Resolver PL 2D via pontos extremos
def resolve_lp_2d(A, b, c):
    m, n = A.shape
    assert n == 2, "Este resolvedor assume 2 variáveis (x, y)."

    linhas = [(A[i], b[i]) for i in range(m)]
    candidatos = []

    # Interseções de todos os pares de restrições (como igualdades)
    for i, j in combinations(range(m), 2):
        p = intersecao(linhas[i], linhas[j])
        if p is None:
            continue
        # Filtra viabilidade: A @ p <= b + EPS
        if np.all(A @ p <= b + EPS):
            candidatos.append(p)

    # Garante origem como fallback (útil se for viável)
    candidatos.append(np.array([0.0, 0.0]))

    P = np.array(candidatos)
    valores = P @ c
    k = int(np.argmax(valores))
    return P[k], float(valores[k]), P, valores


#  TESTES 
def _testes():
    print(">> Rodando testes...")
    # 1) Checa shapes e dados básicos
    assert A.shape == (4, 2), "A deve ser 4x2"
    assert b.shape == (4,), "b deve ter 4 elementos"
    assert c.shape == (2,), "c deve ter 2 elementos"

    # 2) Resolve e confere ótimo esperado
    x_opt, z_opt, P, V = resolve_lp_2d(A, b, c)
    print(f"Ótimo encontrado: x*={x_opt}, z*={z_opt:.3f}")

    # Ótimo esperado para o problema dado: (x, y) = (40, 20), z = 220
    assert np.allclose(x_opt, [40, 20], atol=1e-6), "solução ótima esperada é (40, 20)"
    assert np.isclose(z_opt, 220.0, atol=1e-6), "valor ótimo esperado é 220.0"

    # 3) Checa que pontos clássicos aparecem como candidatos viáveis
    def viavel(p): return np.all(A @ p <= b + EPS)
    assert viavel(np.array([50, 0])), "(50,0) deve ser viável"
    assert viavel(np.array([0, 40])), "(0,40) deve ser viável"

    print(" Passou nos testes!")

_testes()
