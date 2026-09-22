# Seguridad

Este proyecto contiene un modelo serializado con `joblib`. Estos archivos se basan en pickle y solo deben cargarse cuando provengan de una fuente confiable. La aplicación verifica el hash del artefacto incluido y no permite que un usuario cargue modelos externos.

El repositorio debe permanecer privado. No se deben incorporar espectros, resultados de laboratorio, bases de entrenamiento, credenciales ni archivos de configuración con secretos.

Los despliegues externos deben cumplir las reglas corporativas de clasificación, transmisión y almacenamiento de información.

