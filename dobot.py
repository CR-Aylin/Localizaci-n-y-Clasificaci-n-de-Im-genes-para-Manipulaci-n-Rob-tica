#!/usr/bin/env python3
"""
Movimiento rectangular con Dobot Magician

Recorre un rectángulo en el plano XY manteniendo Z constante.
"""

import time
from serial.tools import list_ports
from pydobot import Dobot


def conectar():
    puertos = list(list_ports.comports())

    if len(puertos) == 0:
        raise Exception("No se encontró ningún puerto serie.")

    print("Puertos encontrados:")

    for p in puertos:
        print(f"  {p.device} - {p.description}")

    puerto = puertos[-1].device
    print(f"\nConectando a {puerto}...\n")

    robot = Dobot(port=puerto, verbose=False)

    return robot


def mover_rectangulo(robot,
                     ancho=80,
                     alto=50,
                     velocidad=1.0,
                     repeticiones=3):

    x, y, z, r, *_ = robot.pose()

    print("Posición inicial")
    print(f"X={x:.1f}")
    print(f"Y={y:.1f}")
    print(f"Z={z:.1f}")

    esquinas = [
        (x,          y),
        (x+ancho,    y),
        (x+ancho,    y+alto),
        (x,          y+alto),
        (x,          y)
    ]

    for n in range(repeticiones):

        print(f"\nVuelta {n+1}")

        for px, py in esquinas:

            print(f"Moviendo a ({px:.1f}, {py:.1f}, {z:.1f})")

            # wait=True espera a que termine el movimiento
            robot.move_to(px, py, z, r, wait=True)

            time.sleep(velocidad)


def main():

    robot = conectar()

    try:

        mover_rectangulo(
            robot,
            ancho=80,      # mm
            alto=50,       # mm
            velocidad=0.3, # s entre vértices
            repeticiones=5
        )

    finally:

        print("\nRegresando al punto inicial...")

        x, y, z, r, *_ = robot.pose()

        robot.move_to(x, y, z, r, wait=True)

        robot.close()

        print("Conexión cerrada.")


if __name__ == "__main__":
    main()