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

"""
Resposta: Normalmente o cabeçalho é somente uma descrição que contem o nome, data e versão
do código sem nenhuma instrução direta, já o DocString é a instrução de como funciona o código.

"""

###  abaixo segue desafio... 

## dica: pesquisa operacional...

import numpy as np
from itertools import combinations

EPS = 1e-9

#  1/3) Modele A, b, c 
# Forma padrão: A @ [x,y] <= b, com não-negatividade via -I @ [x,y] <= 0
# Restrição 1: 2x +  y <= 100 -> 2 e 1 vão para A e o 100 para B
# Restrição 2:  x + 2y <= 80 -> 1 e 2 vão para A e 80 para B
# Restrição 3:  x >= 0  ->  -x <= 0 -> mesma coisa -> -1,0 <=0 -> -1,0 para A e 0 para B
# Restrição 4:  y >= 0  ->  -y <= 0 -> mesma coisa -> 0,-1 <=0 -> 0,-1 para A e 0 para B

#guarda os numeros que se multiplicam
A = np.array([
    [2,1], #2x +  y
    [1,2], # x + 2y
    [-1,0], # -x <= 0
    [0,-1] # -y <= 0
], dtype=float)

#guarda os limites
b = np.array([
    100,
    80,
    0,
    0
], dtype=float)

# Função objetivo: z = 3x + 5y
c = np.array([
    3,5
], dtype=float)


#  2/3) Interseção de duas retas
# Cada reta é dada por (a, beta) representando a @ [x,y] = beta
def intersecao(reta1, reta2):
    a1, b1 = reta1
    a2, b2 = reta2
    Aeq = np.vstack([a1, a2])        # 2x2 - Aeq é a direção - singular
    beq = np.array([b1, b2], float)  # 2, - Beq posição inicial
    # Se forem paralelas/singulares, retorne None
    # Dica: use det ou tente resolver e capture exceção
    # usar: np.linalg.det() - para saber se as retas são paralelas
    # usar: np.linalg.solve() - me da as coordenadas x e y do cruzamento
    # TODO:
    dete = np.linalg.det(Aeq)
    if np.abs(dete) < EPS: #abs é o valor absoluto e o EPS(épsilon) é o numero muito pequeno
        return None

    inter = np.linalg.solve(Aeq,beq)
    return inter
    

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
