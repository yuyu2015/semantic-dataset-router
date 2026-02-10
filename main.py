import typer
import sys
import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel

# 初始化 Typer 应用和 Rich 控制台
app = typer.Typer()
console = Console()

@app.command()
def hello():
    """
    一个简单的环境测试命令。
    """
    # 1. 打印一个漂亮的标题
    console.print(Panel.fit("🚀 Semantic Dataset Router", style="bold magenta"))
    
    # 2. 验证 Python 版本
    console.print(f"✅ [bold]Python Executable:[/bold] {sys.executable}")
    console.print(f"✅ [bold]Python Version:[/bold] [green]{sys.version.split()[0]}[/green]")

    # 3. 验证依赖包版本
    console.print(f"✅ [bold]Pandas Version:[/bold] [cyan]{pd.__version__}[/cyan]")
    console.print(f"✅ [bold]Numpy Version:[/bold] [cyan]{np.__version__}[/cyan]")

    # 4. 打印成功消息
    console.print("\n[bold green]Environment is ready! Let's build something awesome.[/bold green] ✨")

if __name__ == "__main__":
    app()
