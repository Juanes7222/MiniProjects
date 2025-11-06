import argparse
import os
from pathlib import Path
from cryptography.fernet import Fernet
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Generate a key and save it to a file
def generate_password(key_path="sources/password.key"):
    """Generate a new encryption key and save it"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating encryption key...", total=None)
        
        password = Fernet.generate_key()
        Path(key_path).parent.mkdir(parents=True, exist_ok=True)
        with open(key_path, "wb") as file_password:
            file_password.write(password)
        
        progress.update(task, completed=True)
    
    console.print(Panel(
        f"[green]✓[/green] Key generated and saved to: [bold cyan]{key_path}[/bold cyan]",
        border_style="green"
    ))

# Load the key
def load_password(key_path="sources/password.key"):
    """Load the encryption key from file"""
    if not os.path.exists(key_path):
        console.print(f"[red]✗ Error:[/red] Key file not found: [yellow]{key_path}[/yellow]")
        console.print("[dim]Generate one first with --generate-key[/dim]")
        raise FileNotFoundError(f"Key file not found: {key_path}")
    return open(key_path, "rb").read()

def create_Fernet(key_path="sources/password.key"):
    """Create a Fernet instance with the loaded key"""
    return Fernet(load_password(key_path))

def crypt_data(data, key_path="sources/password.key"):
    """Encrypt data using Fernet"""
    f = create_Fernet(key_path)
    encrypted_data = f.encrypt(data)
    return encrypted_data

def save_crypt(encrypted_data, file_name):
    """Save encrypted data to a file"""
    with open(file_name + ".enc", "wb") as crypt_file:
        crypt_file.write(encrypted_data)

# Encrypt the file
def encrypt_file(file_name, key_path="sources/password.key"):
    """Encrypt a file and save it with .enc extension"""
    if not os.path.exists(file_name):
        console.print(f"[red]  Error:[/red] File not found: [yellow]{file_name}[/yellow]")
        raise FileNotFoundError(f"File not found: {file_name}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Encrypting [cyan]{os.path.basename(file_name)}[/cyan]...", total=None)
        
        with open(file_name, "rb") as file:
            data_file = file.read()
        
        encrypted_data = crypt_data(data_file, key_path)
        save_crypt(encrypted_data, file_name)
        
        progress.update(task, completed=True)
    
    file_size = os.path.getsize(file_name + ".enc")
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan")
    table.add_column(style="white")
    table.add_row("Original file:", file_name)
    table.add_row("Encrypted file:", f"{file_name}.enc")
    table.add_row("Size:", f"{file_size:,} bytes")
    
    console.print(Panel(
        table,
        title="[green]  File Encrypted Successfully[/green]",
        border_style="green"
    ))

def decrypt_file(encrypted_file_name, output_file=None, key_path="sources/password.key"):
    """Decrypt an encrypted file"""
    if not os.path.exists(encrypted_file_name):
        console.print(f"[red]  Error:[/red] Encrypted file not found: [yellow]{encrypted_file_name}[/yellow]")
        raise FileNotFoundError(f"Encrypted file not found: {encrypted_file_name}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Decrypting [cyan]{os.path.basename(encrypted_file_name)}[/cyan]...", total=None)
        
        f = create_Fernet(key_path)
        
        with open(encrypted_file_name, "rb") as crypt_file:
            encrypted_data = crypt_file.read()
        
        decrypted_data = f.decrypt(encrypted_data)
        
        # Save decrypted data if output file is specified
        if output_file:
            with open(output_file, "wb") as file:
                file.write(decrypted_data)
        
        progress.update(task, completed=True)
    
    if output_file:
        file_size = os.path.getsize(output_file)
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan")
        table.add_column(style="white")
        table.add_row("Encrypted file:", encrypted_file_name)
        table.add_row("Decrypted file:", output_file)
        table.add_row("Size:", f"{file_size:,} bytes")
        
        console.print(Panel(
            table,
            title="[green]  File Decrypted Successfully[/green]",
            border_style="green"
        ))
    
    return decrypted_data

def main():
    parser = argparse.ArgumentParser(
        description="Encrypt and decrypt files using Fernet symmetric encryption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a new encryption key
  %(prog)s --generate-key
  
  # Encrypt a file
  %(prog)s --encrypt myfile.txt
  
  # Decrypt a file
  %(prog)s --decrypt myfile.txt.enc --output myfile_decrypted.txt
  
  # Use custom key file location
  %(prog)s --encrypt myfile.txt --key-path /path/to/custom.key
        """
    )
    
    parser.add_argument(
        '--generate-key',
        action='store_true',
        help='Generate a new encryption key'
    )
    
    parser.add_argument(
        '-e', '--encrypt',
        type=str,
        metavar='FILE',
        help='File to encrypt'
    )
    
    parser.add_argument(
        '-d', '--decrypt',
        type=str,
        metavar='FILE',
        help='File to decrypt'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        metavar='FILE',
        help='Output file for decrypted data'
    )
    
    parser.add_argument(
        '-k', '--key-path',
        type=str,
        default='sources/password.key',
        help='Path to the encryption key file (default: sources/password.key)'
    )
    
    args = parser.parse_args()
    
    try:
        console.print()  # Empty line for spacing
        
        # Generate key
        if args.generate_key:
            generate_password(args.key_path)
            return
        
        # Encrypt file
        if args.encrypt:
            encrypt_file(args.encrypt, args.key_path)
            return
        
        # Decrypt file
        if args.decrypt:
            if not args.output:
                # Default output: remove .enc extension if present
                if args.decrypt.endswith('.enc'):
                    args.output = args.decrypt[:-4]
                else:
                    args.output = args.decrypt + '.decrypted'
            
            decrypt_file(args.decrypt, args.output, args.key_path)
            return
        
        # No action specified
        parser.print_help()
        
    except FileNotFoundError:
        pass  # Error already printed by the function
    except Exception as e:
        console.print(Panel(
            f"[red]  Unexpected Error:[/red]\n{str(e)}",
            border_style="red"
        ))

if __name__ == "__main__":
    main()