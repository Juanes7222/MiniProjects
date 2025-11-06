"""
excel_export_high_quality.py
Exportación de tablas Excel a imagen PNG de alta calidad con interfaz CLI
Requisitos: Windows + Excel instalado
pip install pywin32 pdf2image pillow rich
Nota: pdf2image requiere poppler (descargar de https://github.com/oschwartz10612/poppler-windows/releases/)

Uso:
    python excel_export_high_quality.py archivo.xlsx "Nombre Hoja" salida.png
    python excel_export_high_quality.py archivo.xlsx "Hoja1" salida.png --metodo pdf --dpi 600
    python excel_export_high_quality.py archivo.xlsx "Hoja1" salida.png --scale 5.0 --padding-rows 2
"""

import os
import sys
import time
import argparse
from random import randint
import win32com.client as win32
from win32com.client import constants
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

console = Console()


def parse_args():
    """Configuración de argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Exporta tablas de Excel a imágenes PNG de alta calidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s archivo.xlsx "Hoja1" salida.png
  %(prog)s archivo.xlsx "Datos" salida.png --metodo pdf --dpi 600
  %(prog)s archivo.xlsx "Tabla" salida.png --scale 5.0 --padding-rows 2 --padding-cols 1
  %(prog)s archivo.xlsx "Ventas" salida.png -m chart -s 4.5

Métodos de exportación:
  chart - Rápido y buena calidad (recomendado para uso diario)
  pdf   - Máxima calidad pero más lento (ideal para documentos finales)
        """
    )
    
    # Argumentos posicionales (obligatorios)
    parser.add_argument(
        'excel',
        help='Ruta del archivo Excel (.xlsx, .xls)'
    )
    parser.add_argument(
        'hoja',
        help='Nombre de la hoja a exportar'
    )
    parser.add_argument(
        'salida',
        help='Ruta del archivo PNG de salida'
    )
    
    # Argumentos opcionales
    parser.add_argument(
        '-m', '--metodo',
        choices=['chart', 'pdf'],
        default='chart',
        help='Método de exportación (default: chart)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=450,
        metavar='N',
        help='DPI para método PDF (300-600 recomendado, default: 450)'
    )
    parser.add_argument(
        '-s', '--scale',
        type=float,
        default=4.5,
        metavar='F',
        help='Factor de escala para método Chart (3.0-5.0 recomendado, default: 4.5)'
    )
    parser.add_argument(
        '--padding-rows',
        type=int,
        default=0,
        metavar='N',
        help='Filas de padding adicionales arriba/abajo (default: 0)'
    )
    parser.add_argument(
        '--padding-cols',
        type=int,
        default=0,
        metavar='N',
        help='Columnas de padding adicionales izquierda/derecha (default: 0)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Modo silencioso (solo muestra errores)'
    )
    parser.add_argument(
        '--no-postprocess',
        action='store_true',
        help='Desactivar post-procesamiento de imagen (nitidez, contraste)'
    )
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s 2.0'
    )
    
    args = parser.parse_args()
    
    # Validaciones
    if not os.path.exists(args.excel):
        parser.error(f"El archivo '{args.excel}' no existe")
    
    if not args.excel.lower().endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
        parser.error(f"El archivo debe ser un archivo Excel válido (.xlsx, .xls, .xlsm, .xlsb)")
    
    if not args.salida.lower().endswith('.png'):
        parser.error(f"El archivo de salida debe tener extensión .png")
    
    if args.dpi < 72 or args.dpi > 2400:
        parser.error(f"DPI debe estar entre 72 y 2400 (recibido: {args.dpi})")
    
    if args.scale < 1.0 or args.scale > 10.0:
        parser.error(f"Scale factor debe estar entre 1.0 y 10.0 (recibido: {args.scale})")
    
    if args.padding_rows < 0 or args.padding_cols < 0:
        parser.error("Los valores de padding no pueden ser negativos")
    
    return args


def abrir_excel():
    """Abre Excel de forma invisible"""
    excel = win32.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    return excel


def texto_de_celda(ws, r, c):
    """Obtiene el texto visible de una celda"""
    try:
        return str(ws.Cells(r, c).Text).strip()
    except:
        try:
            val = ws.Cells(r, c).Value
            return "" if val is None else str(val).strip()
        except:
            return ""


def detectar_por_listobject(ws):
    """Detecta tabla usando ListObject de Excel"""
    try:
        lo_count = ws.ListObjects().Count
    except:
        lo_count = 0
    
    if lo_count >= 1:
        lo = ws.ListObjects(1)
        rng = lo.Range
        return rng.Row, rng.Row + rng.Rows.Count - 1, rng.Column, rng.Column + rng.Columns.Count - 1
    return None


def detectar_por_usedrange_text(ws):
    """Detecta límites de tabla escaneando celdas con contenido"""
    ur = ws.UsedRange
    r0 = ur.Row
    c0 = ur.Column
    max_r = r0 + ur.Rows.Count - 1
    max_c = c0 + ur.Columns.Count - 1

    fila_ini = None
    fila_fin = None
    col_ini = None
    col_fin = None

    # Buscar primera y última fila con contenido
    for r in range(r0, max_r + 1):
        if any(texto_de_celda(ws, r, c) != "" for c in range(c0, max_c + 1)):
            fila_ini = r
            break

    for r in range(max_r, r0 - 1, -1):
        if any(texto_de_celda(ws, r, c) != "" for c in range(c0, max_c + 1)):
            fila_fin = r
            break

    # Buscar primera y última columna con contenido
    for c in range(c0, max_c + 1):
        if any(texto_de_celda(ws, r, c) != "" for r in range(r0, max_r + 1)):
            col_ini = c
            break

    for c in range(max_c, c0 - 1, -1):
        if any(texto_de_celda(ws, r, c) != "" for r in range(r0, max_r + 1)):
            col_fin = c
            break

    if fila_ini is None or fila_fin is None or col_ini is None or col_fin is None:
        return None

    return fila_ini, fila_fin, col_ini, col_fin


def ajustar_por_merged(ws, fila_ini, fila_fin, col_ini, col_fin, progress=None, task=None):
    """Expande el rango para incluir celdas combinadas completas"""
    changed = True
    iterations = 0
    total_cells = (fila_fin - fila_ini + 1) * (col_fin - col_ini + 1)
    processed = 0
    
    while changed:
        changed = False
        iterations += 1
        for r in range(fila_ini, fila_fin + 1):
            for c in range(col_ini, col_fin + 1):
                processed += 1
                if progress and task:
                    progress.update(task, completed=min(processed * 100 // total_cells, 95))
                
                try:
                    cell = ws.Cells(r, c)
                    if cell.MergeCells:
                        ma = cell.MergeArea
                        ma_r1 = ma.Row
                        ma_r2 = ma.Row + ma.Rows.Count - 1
                        ma_c1 = ma.Column
                        ma_c2 = ma.Column + ma.Columns.Count - 1
                        
                        if ma_r1 < fila_ini:
                            fila_ini = ma_r1
                            changed = True
                        if ma_r2 > fila_fin:
                            fila_fin = ma_r2
                            changed = True
                        if ma_c1 < col_ini:
                            col_ini = ma_c1
                            changed = True
                        if ma_c2 > col_fin:
                            col_fin = ma_c2
                            changed = True
                except:
                    pass
    
    if progress and task:
        progress.update(task, completed=100)
    
    return fila_ini, fila_fin, col_ini, col_fin


def mejorar_calidad_imagen(img, progress=None, task=None):
    """Post-procesamiento para mejorar nitidez y contraste"""
    if progress and task:
        progress.update(task, completed=0, description="[cyan]Aplicando nitidez...")
    
    # Aplicar nitidez (UnsharpMask)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    
    if progress and task:
        progress.update(task, completed=33, description="[cyan]Mejorando contraste...")
    
    # Mejorar contraste sutilmente
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.05)
    
    if progress and task:
        progress.update(task, completed=66, description="[cyan]Afinando detalles...")
    
    # Mejorar nitidez adicional
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    if progress and task:
        progress.update(task, completed=100)
    
    return img


def exportar_via_pdf_vectorial(ws, fila_ini, fila_fin, col_ini, col_fin, ruta_salida, dpi=450, quiet=False, postprocess=True):
    """
    Exportación vía PDF vectorial + rasterización de alta calidad
    Mejor para: tablas con mucho texto, máxima nitidez
    """
    if quiet:
        # Modo silencioso: exportar sin barras de progreso
        wb = ws.Parent
        rng = ws.Range(ws.Cells(fila_ini, col_ini), ws.Cells(fila_fin, col_fin))
        temp = wb.Worksheets.Add(After=wb.Worksheets(wb.Worksheets.Count))
        
        try:
            rng.Copy()
            temp.Range("A1").PasteSpecial(-4163)
            
            ncols = rng.Columns.Count
            nrows = rng.Rows.Count
            
            for i in range(ncols):
                try:
                    temp.Columns(i + 1).ColumnWidth = ws.Columns(col_ini + i).ColumnWidth
                except:
                    pass
            
            for i in range(nrows):
                try:
                    temp.Rows(i + 1).RowHeight = ws.Rows(fila_ini + i).RowHeight
                except:
                    pass
            
            ps = temp.PageSetup
            ps.LeftMargin = 0
            ps.RightMargin = 0
            ps.TopMargin = 0
            ps.BottomMargin = 0
            ps.HeaderMargin = 0
            ps.FooterMargin = 0
            ps.Zoom = False
            ps.FitToPagesWide = 1
            ps.FitToPagesTall = 1
            
            ruta_pdf = os.path.splitext(os.path.abspath(ruta_salida))[0] + "_temp.pdf"
            temp.ExportAsFixedFormat(0, ruta_pdf, Quality=0)
            
        finally:
            try:
                temp.Delete()
            except:
                pass
        
        images = convert_from_path(ruta_pdf, dpi=dpi, fmt='png')
        img = images[0]
        
        if postprocess:
            img = mejorar_calidad_imagen(img)
        
        img.save(ruta_salida, format="PNG", optimize=True, compress_level=3)
        
        try:
            os.remove(ruta_pdf)
        except:
            pass
        
        return ruta_salida
    
    # Modo normal con barras de progreso
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[yellow]Preparando exportación PDF...", total=100)
        
        wb = ws.Parent
        rng = ws.Range(ws.Cells(fila_ini, col_ini), ws.Cells(fila_fin, col_fin))
        temp = wb.Worksheets.Add(After=wb.Worksheets(wb.Worksheets.Count))
        
        try:
            progress.update(task, completed=10, description="[yellow]Copiando rango...")
            
            rng.Copy()
            temp.Range("A1").PasteSpecial(-4163)
            
            progress.update(task, completed=25, description="[yellow]Ajustando dimensiones...")
            
            ncols = rng.Columns.Count
            nrows = rng.Rows.Count
            
            for i in range(ncols):
                try:
                    temp.Columns(i + 1).ColumnWidth = ws.Columns(col_ini + i).ColumnWidth
                except:
                    pass
            
            for i in range(nrows):
                try:
                    temp.Rows(i + 1).RowHeight = ws.Rows(fila_ini + i).RowHeight
                except:
                    pass
            
            progress.update(task, completed=40, description="[yellow]Configurando página...")
            
            ps = temp.PageSetup
            ps.LeftMargin = 0
            ps.RightMargin = 0
            ps.TopMargin = 0
            ps.BottomMargin = 0
            ps.HeaderMargin = 0
            ps.FooterMargin = 0
            ps.Zoom = False
            ps.FitToPagesWide = 1
            ps.FitToPagesTall = 1
            
            progress.update(task, completed=50, description="[yellow]Exportando a PDF...")
            
            ruta_pdf = os.path.splitext(os.path.abspath(ruta_salida))[0] + "_temp.pdf"
            temp.ExportAsFixedFormat(0, ruta_pdf, Quality=0)
            
            progress.update(task, completed=70, description=f"[yellow]Rasterizando a {dpi} DPI...")
            
        finally:
            try:
                temp.Delete()
            except:
                pass
        
        images = convert_from_path(ruta_pdf, dpi=dpi, fmt='png')
        img = images[0]
        
        if postprocess:
            progress.update(task, completed=85, description="[yellow]Post-procesando imagen...")
            img = mejorar_calidad_imagen(img)
        
        progress.update(task, completed=95, description="[yellow]Guardando imagen...")
        
        img.save(ruta_salida, format="PNG", optimize=True, compress_level=3)
        
        try:
            os.remove(ruta_pdf)
        except:
            pass
        
        progress.update(task, completed=100, description="[green]✓ Exportación PDF completada")
        time.sleep(0.3)
    
    return ruta_salida


def exportar_via_chart_mejorado(ws, fila_ini, fila_fin, col_ini, col_fin, ruta_salida, scale_factor=4.5, quiet=False, postprocess=True):
    """
    Exportación vía Chart con alta escala
    Mejor para: balance entre calidad y velocidad
    """
    if quiet:
        # Modo silencioso
        rng = ws.Range(ws.Cells(fila_ini, col_ini), ws.Cells(fila_fin, col_fin))
        
        try:
            rng.CopyPicture(Appearance=constants.xlPrinter, Format=-4147)
        except:
            try:
                rng.CopyPicture(Appearance=2, Format=-4147)
            except:
                rng.CopyPicture()
        
        time.sleep(0.2)
        
        base_w = max(10, rng.Width)
        base_h = max(10, rng.Height)
        chart_w = int(base_w * scale_factor)
        chart_h = int(base_h * scale_factor)
        
        left_pos = 10 + randint(0, 50)
        top_pos = 10 + randint(0, 50)
        
        chart_obj = ws.ChartObjects().Add(left_pos, top_pos, chart_w, chart_h)
        chart = chart_obj.Chart
        
        for intento in range(3):
            try:
                chart.Paste()
                break
            except:
                time.sleep(0.15)
        
        try:
            chart_obj.Width = chart_w
            chart_obj.Height = chart_h
            chart.ChartArea.Width = chart_w
            chart.ChartArea.Height = chart_h
        except:
            pass
        
        ruta_abs = os.path.abspath(ruta_salida)
        os.makedirs(os.path.dirname(ruta_abs), exist_ok=True)
        
        export_ok = False
        for intento in range(5):
            try:
                chart.Export(ruta_abs, FilterName="PNG")
                export_ok = True
                break
            except:
                time.sleep(0.2)
        
        try:
            chart_obj.Delete()
        except:
            pass
        
        if not export_ok:
            raise RuntimeError("No se pudo exportar el chart desde Excel.")
        
        if postprocess:
            try:
                img = Image.open(ruta_abs)
                img = mejorar_calidad_imagen(img)
                img.save(ruta_abs, format="PNG", optimize=True, compress_level=3)
            except:
                pass
        
        return ruta_abs
    
    # Modo normal con barras de progreso
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Preparando exportación Chart...", total=100)
        
        rng = ws.Range(ws.Cells(fila_ini, col_ini), ws.Cells(fila_fin, col_fin))
        
        progress.update(task, completed=10, description="[cyan]Copiando rango al portapapeles...")
        
        try:
            rng.CopyPicture(Appearance=constants.xlPrinter, Format=-4147)
        except:
            try:
                rng.CopyPicture(Appearance=2, Format=-4147)
            except:
                rng.CopyPicture()
        
        time.sleep(0.2)
        
        progress.update(task, completed=25, description="[cyan]Calculando dimensiones...")
        
        base_w = max(10, rng.Width)
        base_h = max(10, rng.Height)
        chart_w = int(base_w * scale_factor)
        chart_h = int(base_h * scale_factor)
        
        progress.update(task, completed=35, description="[cyan]Creando gráfico temporal...")
        
        left_pos = 10 + randint(0, 50)
        top_pos = 10 + randint(0, 50)
        
        chart_obj = ws.ChartObjects().Add(left_pos, top_pos, chart_w, chart_h)
        chart = chart_obj.Chart
        
        progress.update(task, completed=45, description="[cyan]Pegando imagen en gráfico...")
        
        for intento in range(3):
            try:
                chart.Paste()
                break
            except:
                time.sleep(0.15)
        
        progress.update(task, completed=55, description="[cyan]Ajustando tamaño...")
        
        try:
            chart_obj.Width = chart_w
            chart_obj.Height = chart_h
            chart.ChartArea.Width = chart_w
            chart.ChartArea.Height = chart_h
        except:
            pass
        
        progress.update(task, completed=65, description="[cyan]Exportando a PNG...")
        
        ruta_abs = os.path.abspath(ruta_salida)
        os.makedirs(os.path.dirname(ruta_abs), exist_ok=True)
        
        export_ok = False
        for intento in range(5):
            try:
                chart.Export(ruta_abs, FilterName="PNG")
                export_ok = True
                break
            except:
                time.sleep(0.2)
        
        progress.update(task, completed=75, description="[cyan]Limpiando recursos...")
        
        try:
            chart_obj.Delete()
        except:
            pass
        
        if not export_ok:
            raise RuntimeError("No se pudo exportar el chart desde Excel.")
        
        if postprocess:
            progress.update(task, completed=85, description="[cyan]Post-procesando imagen...")
            
            try:
                img = Image.open(ruta_abs)
                img = mejorar_calidad_imagen(img)
                img.save(ruta_abs, format="PNG", optimize=True, compress_level=3)
            except Exception as e:
                console.print(f"[yellow]⚠ Advertencia: no se pudo post-procesar: {e}[/yellow]")
        
        progress.update(task, completed=100, description="[green]✓ Exportación Chart completada")
        time.sleep(0.3)
    
    return ruta_abs


def exportar_rango_alta_calidad(ws, fila_ini, fila_fin, col_ini, col_fin, ruta_salida, 
                                 metodo='chart', dpi=450, scale_factor=4.5, quiet=False, postprocess=True):
    """
    Exporta rango de Excel a PNG de alta calidad
    
    metodo='pdf': Máxima calidad, más lento (requiere poppler)
    metodo='chart': Buena calidad, más rápido (recomendado)
    """
    if metodo == 'pdf':
        return exportar_via_pdf_vectorial(ws, fila_ini, fila_fin, col_ini, col_fin, 
                                          ruta_salida, dpi=dpi, quiet=quiet, postprocess=postprocess)
    elif metodo == 'chart':
        return exportar_via_chart_mejorado(ws, fila_ini, fila_fin, col_ini, col_fin, 
                                           ruta_salida, scale_factor=scale_factor, quiet=quiet, postprocess=postprocess)
    else:
        raise ValueError(f"Método '{metodo}' no válido. Usa 'pdf' o 'chart'")


def main():
    args = parse_args()
    
    if not args.quiet:
        # Banner inicial
        console.print()
        console.print(Panel.fit(
            "[bold cyan]EXPORTADOR DE TABLAS EXCEL[/bold cyan]\n"
            "[white]Alta calidad • Rápido • Automático[/white]",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()
        
        # Tabla de configuración
        config_table = Table(title="⚙️  Configuración", box=box.ROUNDED, show_header=False, border_style="blue")
        config_table.add_column("Parámetro", style="cyan", width=20)
        config_table.add_column("Valor", style="yellow")
        
        config_table.add_row("📁 Archivo", os.path.basename(args.excel))
        config_table.add_row("📄 Hoja", args.hoja)
        config_table.add_row("🎨 Método", args.metodo.upper())
        if args.metodo == 'pdf':
            config_table.add_row("🔍 DPI", str(args.dpi))
        else:
            config_table.add_row("📏 Scale Factor", str(args.scale))
        config_table.add_row("💾 Salida", os.path.basename(args.salida))
        if args.padding_rows > 0 or args.padding_cols > 0:
            config_table.add_row("📐 Padding", f"{args.padding_rows} filas, {args.padding_cols} cols")
        if args.no_postprocess:
            config_table.add_row("⚠️  Post-proceso", "Desactivado")
        
        console.print(config_table)
        console.print()
    
    try:
        # Paso 1: Abrir Excel
        if not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[green]Abriendo Excel...", total=None)
                excel = abrir_excel()
                wb = excel.Workbooks.Open(os.path.abspath(args.excel))
                ws = wb.Worksheets(args.hoja)
                time.sleep(0.2)
                progress.update(task, description="[green]✓ Excel abierto correctamente")
                time.sleep(0.3)
            console.print()
        else:
            excel = abrir_excel()
            wb = excel.Workbooks.Open(os.path.abspath(args.excel))
            ws = wb.Worksheets(args.hoja)
        
        # Paso 2: Detectar rango
        if not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[magenta]Detectando rango de tabla...", total=None)
                
                res = detectar_por_listobject(ws)
                
                if res:
                    fila_ini, fila_fin, col_ini, col_fin = res
                    metodo_deteccion = "ListObject"
                else:
                    res2 = detectar_por_usedrange_text(ws)
                    if not res2:
                        wb.Close(SaveChanges=False)
                        excel.Quit()
                        raise RuntimeError("No se pudo detectar rango: la hoja parece vacía.")
                    fila_ini, fila_fin, col_ini, col_fin = res2
                    metodo_deteccion = "UsedRange + Análisis"
                
                time.sleep(0.2)
                progress.update(task, description=f"[green]✓ Rango detectado ({metodo_deteccion})")
                time.sleep(0.3)
            console.print()
        else:
            res = detectar_por_listobject(ws)
            if res:
                fila_ini, fila_fin, col_ini, col_fin = res
            else:
                res2 = detectar_por_usedrange_text(ws)
                if not res2:
                    wb.Close(SaveChanges=False)
                    excel.Quit()
                    raise RuntimeError("No se pudo detectar rango: la hoja parece vacía.")
                fila_ini, fila_fin, col_ini, col_fin = res2
        
        # Paso 3: Ajustar por celdas combinadas
        if not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("[blue]Procesando celdas combinadas...", total=100)
                fila_ini, fila_fin, col_ini, col_fin = ajustar_por_merged(
                    ws, fila_ini, fila_fin, col_ini, col_fin, progress, task
                )
                progress.update(task, description="[green]✓ Celdas combinadas procesadas")
                time.sleep(0.3)
            console.print()
        else:
            fila_ini, fila_fin, col_ini, col_fin = ajustar_por_merged(
                ws, fila_ini, fila_fin, col_ini, col_fin
            )
        
        # Aplicar padding
        if args.padding_rows > 0 or args.padding_cols > 0:
            max_row = ws.Rows.Count
            max_col = ws.Columns.Count
            fila_ini = max(1, fila_ini - args.padding_rows)
            fila_fin = min(max_row, fila_fin + args.padding_rows)
            col_ini = max(1, col_ini - args.padding_cols)
            col_fin = min(max_col, col_fin + args.padding_cols)
            if not args.quiet:
                console.print(f"[dim]✓ Padding aplicado: {args.padding_rows} filas, {args.padding_cols} columnas[/dim]")
                console.print()
        
        # Mostrar información del rango
        if not args.quiet:
            info_table = Table(title="📊 Información del Rango", box=box.ROUNDED, show_header=False, border_style="green")
            info_table.add_column("Campo", style="cyan", width=20)
            info_table.add_column("Valor", style="white")
            
            info_table.add_row("Filas", f"{fila_ini} → {fila_fin} ({fila_fin - fila_ini + 1} filas)")
            info_table.add_row("Columnas", f"{col_ini} → {col_fin} ({col_fin - col_ini + 1} columnas)")
            info_table.add_row("Total de celdas", f"{(fila_fin - fila_ini + 1) * (col_fin - col_ini + 1):,}")
            
            console.print(info_table)
            console.print()
        
        # Paso 4: Exportar
        salida = exportar_rango_alta_calidad(
            ws, fila_ini, fila_fin, col_ini, col_fin, 
            args.salida,
            metodo=args.metodo,
            dpi=args.dpi,
            scale_factor=args.scale,
            quiet=args.quiet,
            postprocess=not args.no_postprocess
        )
        
        if not args.quiet:
            console.print()
        
        # Cerrar Excel
        if not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[dim]Cerrando Excel...", total=None)
                wb.Close(SaveChanges=False)
                excel.Quit()
                time.sleep(0.2)
                progress.update(task, description="[dim]✓ Excel cerrado")
                time.sleep(0.2)
            console.print()
        else:
            wb.Close(SaveChanges=False)
            excel.Quit()
        
        # Resultado final
        if not args.quiet:
            try:
                file_size = os.path.getsize(salida) / 1024
                img = Image.open(salida)
                
                resultado_table = Table(
                    title="✅ EXPORTACIÓN COMPLETADA",
                    title_style="bold green",
                    box=box.DOUBLE_EDGE,
                    border_style="green",
                    show_header=False
                )
                resultado_table.add_column("Campo", style="cyan bold", width=20)
                resultado_table.add_column("Valor", style="white")
                
                resultado_table.add_row("  Archivo", os.path.basename(salida))
                resultado_table.add_row("  Ruta completa", os.path.abspath(salida))
                resultado_table.add_row("  Dimensiones", f"{img.width:,} × {img.height:,} píxeles")
                resultado_table.add_row("  Tamaño", f"{file_size:.1f} KB")
                resultado_table.add_row("  Método usado", args.metodo.upper())
                
                console.print(resultado_table)
            except Exception as e:
                console.print(Panel(
                    f"[green]✓ Archivo exportado:[/green] [white]{os.path.abspath(salida)}[/white]",
                    border_style="green"
                ))
            
            console.print()
        else:
            # En modo silencioso, solo imprimir la ruta del archivo generado
            print(os.path.abspath(salida))
        
    except Exception as e:
        if not args.quiet:
            console.print()
            console.print(Panel(
                f"[bold red]ERROR:[/bold red]\n[white]{str(e)}[/white]",
                border_style="red",
                title="❌ Error en la exportación"
            ))
            import traceback
            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        else:
            print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()