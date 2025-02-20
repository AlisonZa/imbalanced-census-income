import os
from entities.entities import FeatureDefinition, EnviromentConfiguration

class MarkdownLogger:
    """
    A simple Markdown logger that writes text to a .md file using Environment Configuration.
    """

    def __init__(self, environment_configuration: EnviromentConfiguration = EnviromentConfiguration(), title="Statistical Report"):
        self.log_file = environment_configuration.statistical_report  # Use the configured path
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)  # Ensure directory exists
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")  # Add a title at the beginning

    def write(self, message):
        """Writes a plain text message to the markdown file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n\n")

    def section(self, title):
        """Creates a new section with a Markdown header."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n## {title}\n\n")

# Instantiate logger using environment configuration
env_config = EnviromentConfiguration()
markdown_logger = MarkdownLogger(env_config)
