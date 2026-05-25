# Despliegue del Tablero Saber 11 en AWS EC2 con Docker

Guía paso a paso adaptada del **Taller 12 - Docker** para desplegar el
tablero (`app.py`) del Proyecto 2 ACTD en una instancia EC2.

**Nombre del grupo:** `<DEFINIR>` — usarlo como prefijo de todos los recursos.

---

## 0. Pre-requisitos

- Cuenta de AWS con consola accesible
- Región **N. Virginia (us-east-1)** en la esquina superior derecha de la consola
- Llave `.pem` para SSH descargada
- Repositorio del proyecto pusheado a GitHub con la última versión (incluye `models/pregunta_1/`, `Data/saber11_Antioquia_clean.csv`, `mlruns/pregunta_1/`)

---

## 1. Lanzar la instancia EC2

1. Consola → **EC2** → **Launch instance**
2. Nombre: `<grupo>-saber11-tablero`
3. AMI: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
4. Tipo: **`t2.medium`** (4 GB RAM — necesario por TensorFlow; `t2.micro` no alcanza)
5. Key pair: tu llave existente o crea una nueva
6. Network settings → **Edit**:
   - Allow SSH traffic from: **My IP** (puerto 22)
   - Add security group rule:
     - Type: **Custom TCP**, Port: **8050**, Source: **Anywhere (0.0.0.0/0)**
7. Storage: **100 GB gp3** (imagen Docker + dataset + TF ocupan ~5 GB; el resto para holgura)
8. **Launch instance**
9. Anota la **IPv4 pública** (algo como `54.xxx.xxx.xxx`)

---

## 2. Conectarse a la instancia

Desde la carpeta donde está tu `.pem`:

```bash
chmod 400 llave.pem               # (solo Mac/Linux)
ssh -i llave.pem ubuntu@<IP>
```

En Windows con PowerShell: `ssh -i .\llave.pem ubuntu@<IP>` funciona también.

---

## 3. Instalar Docker (idéntico al Taller 12)

```bash
# Limpiar versiones viejas (ignora error si no hay)
sudo apt-get remove docker docker-engine docker.io containerd runc -y

# Actualizar índice
sudo apt-get update

# Dependencias
sudo apt-get install -y ca-certificates curl gnupg

# Llave GPG de Docker
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Repo
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# Verificar
sudo docker run hello-world
```

> Captura la salida de `hello-world` para el reporte.

---

## 4. Traer el código del proyecto

```bash
# Clonar el repo
git clone https://github.com/AlejandroCarvajal49/Proyecto2_ACTD.git
cd Proyecto2_ACTD

# Verificar que los modelos y datos estén
ls -la models/pregunta_1/regresion/best/
ls -la models/pregunta_1/clasificacion_binaria/best/
ls -lah Data/saber11_Antioquia_clean.csv
ls -la despliegue/Dockerfile
```

Deberías ver `model.keras`, `preprocessor.pkl`, `metadata.json` en los directorios de modelos.

> Si el repo no tiene los archivos `models/pregunta_1/` aún, antes de hacer este paso en Windows ejecuta `python -m Analysis.entrenar_modelos_p1` y haz `git add models/pregunta_1/ && git commit && git push`.

---

## 5. Construir la imagen Docker

Desde la raíz del repo (`~/Proyecto2_ACTD`):

```bash
sudo docker build -f despliegue/Dockerfile -t saber11:latest .
```

Esto tarda ~5-10 min la primera vez (descarga TensorFlow ~600 MB).

> Captura el "Successfully tagged saber11:latest" para Soporte 4.

Verifica:
```bash
sudo docker image ls
```

---

## 6. Correr el contenedor

```bash
sudo docker run -d \
    -p 8050:8050 \
    --name saber11 \
    --restart unless-stopped \
    saber11:latest
```

Banderas:
- `-d` detach (corre en background)
- `-p 8050:8050` mapea puerto del host al del contenedor
- `--name saber11` para referirlo fácil
- `--restart unless-stopped` se reinicia solo si reinicias la VM

Verifica que esté corriendo:
```bash
sudo docker ps
sudo docker logs saber11 --tail 20
```

Deberías ver el log de Dash:
```
Dash is running on http://0.0.0.0:8050/
```

---

## 7. Probar en el navegador

Abre:
```
http://<IP-publica-EC2>:8050/pregunta_1
```

Deberías ver el tablero. La primera carga es lenta (~10-15 s) porque TF inicializa los modelos.

> Captura del navegador con la app funcionando → va al Soporte 4 del reporte.

---

## 8. Pegar la URL en el reporte

En `Entregable1-Reporte.docx`, sección **"Ambiente de despliegue y mantenimiento"** > "URL del tablero en ejecución", reemplaza el placeholder:

```
[PEGAR AQUI: http://<IP-publica-EC2>:8050/pregunta_1]
```

Con la URL real, por ejemplo:
```
http://54.123.45.67:8050/pregunta_1
```

---

## 9. Capturas para Soporte 4

Las capturas que necesita el Soporte 4 según el PDF del proyecto:

1. **Consola EC2** mostrando la instancia "Running" con su IPv4 pública
2. **Security Group** con el inbound rule del puerto 8050 abierto
3. **Terminal SSH** mostrando `sudo docker ps` con `saber11` corriendo
4. **Navegador** con el tablero abierto en `http://IP:8050/pregunta_1`

Guárdalas en `Analitica/screenshots_p1/aws_*.png` y pégalas en el Anexo Soporte 4.

---

## Mantenimiento

| Acción | Comando |
|---|---|
| Ver logs | `sudo docker logs saber11 --tail 50 -f` |
| Reiniciar | `sudo docker restart saber11` |
| Detener | `sudo docker stop saber11` |
| Eliminar contenedor | `sudo docker rm -f saber11` |
| Actualizar app | `cd Proyecto2_ACTD && git pull && sudo docker build -f despliegue/Dockerfile -t saber11:latest . && sudo docker rm -f saber11 && sudo docker run -d -p 8050:8050 --name saber11 --restart unless-stopped saber11:latest` |
| Liberar caché | `sudo docker system prune -af` |

---

## Troubleshooting

**Error: "no space left on device" al hacer build**
→ La instancia tiene poco disco. Sube el volumen EBS o ejecuta `sudo docker system prune -af`.

**El contenedor arranca pero el browser muestra "Connection refused"**
→ El security group no tiene abierto el puerto 8050. Consola EC2 → Security groups → Inbound rules → Add rule.

**El tablero tarda mucho en cargar y los modelos no aparecen**
→ Revisa los logs (`sudo docker logs saber11`). Si ves "OOMKilled", la instancia se quedó sin RAM — sube a `t2.large` o `t3.medium`.

**Error al cargar preprocessor.pkl ('SimpleImputer' no tiene '_fill_dtype')**
→ Hay archivos `.pkl` viejos serializados con sklearn distinto. Borra `models/pregunta_1/custom/` y reconstruye el contenedor.

**Cambias el código y los cambios no se ven**
→ El Dockerfile **copia** el código al construir; cambios después del build no se reflejan. Reconstruye y vuelve a correr (ver "Actualizar app" arriba).
