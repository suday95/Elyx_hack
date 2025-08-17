import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import time

console = Console()

def main():
    console.print(Panel.fit("💬 Elyx RAG CLI", style="bold blue"))
    console.print("Type 'quit' or 'exit' to end the session\n")
    
    while True:
        try:
            # Get user input
            question = console.input("[bold green]You:[/] ")
            
            if question.lower() in ('quit', 'exit'):
                break
                
            if not question.strip():
                continue
                
            # Show spinner while processing
            with console.status("[bold]Consulting the team...") as status:
                start_time = time.time()
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": question}
                ).json()
                elapsed = time.time() - start_time
                
            # Display response
            console.print(Panel.fit(
                Markdown(response["answer"]),
                title=f"[bold]{response['role']}[/]",
                subtitle=f"⏱ {elapsed:.2f}s | 📚 Sources: {len(response['sources'])}",
                border_style="dim"
            ))
            
            # Show sources if requested
            if "sources" in response:
                console.print("\n[dim]Sources referenced:[/]")
                for source in response["sources"]:
                    console.print(f"• [dim]{source}[/]")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/]")

if __name__ == "__main__":
    main()