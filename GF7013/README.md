# Comentarios entrega corregida

Se incluyen los códigos para generar las figuras de los informes de la tarea 2 y 3, particularmente para el caso de las preguntas P44 y P3 respectivamente (que fueron corregidos):

- Para ejecutar y generar las figuras de la Tarea 3 se puede ejecutar como `python GF7013\bin\Tarea3\P3\p3.py`  (genera distribuciones 2D apriori y aposteriori para el problema de la recta ortogonal con TMCMC y remuestreo)

  - Otras figuras:

    - `python GF7013\bin\Tarea3\test\test_tmcmc.py` o `python -m GF7013.bin.Tarea3.test.test_tmcmc`
    - `python GF7013\bin\Tarea3\test\test_tmcmc_resampling.py` o `python -m GF7013.bin.Tarea3.test.test_tmcmc_resampling`
- Para ejecutar y generar las figuras de la Tarea 3 se puede ejecutar como: `python GF7013\bin\Tarea3\P3\p3.py`  o `python -m GF7013.bin.Tarea3.P3.p3` (genera distribuciones 2D apriori y aposteriori para el problema de la recta ortogonal con TMCMC y remuestreo)

> Los códigos modificados respecto a la entrega oficial son `p3.py` en la carpeta GF7013/bin/Tarea3/P3, `P44.py` en la carpeta GF7013/bin/Tarea2/P44, `forward.py` en  GF7013/models/ajuste_ortogonal_recta en que se agregó la clase *forward_ensemble* y `pdf_normal.py` en GF7013/probability_functions/pdf donde se agregó la clase  *pdf_normal_Nmodels*, las que permiten trabajar con los ensembles.
