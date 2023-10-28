import os

class Config:
    # Secret key for securing sessions and forms
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

    # MongoDB configuration
    MONGODB_SETTINGS = {
        'db': os.environ.get('MONGO_DB_NAME', 'your-database-name'),
        'host': os.environ.get('MONGO_URI', 'mongodb://localhost:27017/your-database-name')
    }

    # Enable debug mode (set to False in production)
    DEBUG = True

    # Add other configuration options as needed
    # Example:
    # MAIL_SERVER = 'smtp.example.com'
    # MAIL_PORT = 587
    # MAIL_USE_TLS = True
    # MAIL_USERNAME = 'your-email@example.com'
    # MAIL_PASSWORD = 'your-email-password'

# Development configuration
class DevelopmentConfig(Config):
    DEBUG = True

# Production configuration
class ProductionConfig(Config):
    DEBUG = False

# Choose the configuration class based on your environment
# For example, set the environment variable 'FLASK_ENV' to 'production'
# when deploying to a production server
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig  # Default configuration
}

def create_app():
    # Load configuration based on FLASK_ENV environment variable
    app = Flask(__name__)
    env = os.environ.get('FLASK_ENV', 'default')
    app.config.from_object(config_by_name[env])

    # Initialize and configure extensions (if needed)

    return app