# Projeto19
Otimização de logística de armazéns com CVXPY.
Esta é uma tarefa do curso Otimização Discreta- https://www.coursera.org/learn/discrete-optimization/
O arquivo handout.pdf possui a descrição completa do problema e do formato de dados de entrada

Otimização de logística de armazéns de distribuição.
<p align="justify">
O problema fornece a localização x,y de n clientes e m armazéns de distribuição.
Cada cliente possui uma demanda 'Di' a ser atendida, cada armazém possui uma capacidade
máxima de atendimento 'Ci' e custo de operação 'Cti'. O objetivo é otimizar a seleção
de armazéns que devem ficar abertos de forma a atender todos os clientes existentes e
também a seleção de clientes que devem ser atendidos por cada armazém. A capacidade
máxima de atendimento de cada armazém não deve ser superada pelo somatório de demandas
de clientes Di que estão conectados a cada armazém.Deve-se minimizar o somatório de
distâncias de cada armazém até seus clientes atendidos e o somatório de custos de operação
de cada armazém aberto.
</p>
![Recompensa no tempo](https://github.com/rodfloripa/Projeto19/blob/main/warehouse.jpeg?raw=true)

Instalar Bibliotecas cvxpy,otimizador CbC e cbcpy:

https://github.com/coin-or/Cbc,
pip install cbcpy,
pip install cvxpy

Para rodar baixe o diretório data e digite na linha de comando:
python solver_cvxpy.py ./data/fl_16_1
