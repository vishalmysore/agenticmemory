"""
User Authentication Module
Handles user login, registration, and session management
"""

class UserManager:
    """
    Manages user operations including authentication and authorization.
    Integrates with database layer and caching for optimal performance.
    """
    
    def __init__(self, db_connection, cache_client):
        """
        Initialize UserManager with database and cache connections.
        
        Args:
            db_connection: Active database connection object
            cache_client: Redis cache client instance
        """
        self.db = db_connection
        self.cache = cache_client
        self.users = []
        self.session_timeout = 3600  # 1 hour in seconds
    
    def add_user(self, name, email, password):
        """
        Adds a new user to the system with encrypted password.
        
        Args:
            name (str): Full name of the user
            email (str): Email address (must be unique)
            password (str): Plain text password (will be hashed)
            
        Returns:
            User: Newly created user object
            
        Raises:
            ValueError: If email already exists
            ValidationError: If password doesn't meet requirements
        """
        if self.email_exists(email):
            raise ValueError(f"Email {email} already registered")
        
        hashed_password = self.hash_password(password)
        user = User(name, email, hashed_password)
        self.users.append(user)
        
        # Cache the user object
        self.cache.set(f"user:{email}", user, ex=self.session_timeout)
        
        # Persist to database
        self.db.insert_user(user)
        
        return user
    
    def authenticate(self, email, password):
        """
        Authenticates a user with email and password.
        
        Args:
            email (str): User's email address
            password (str): User's password
            
        Returns:
            tuple: (success: bool, session_token: str or None)
        """
        user = self.get_user_by_email(email)
        if not user:
            return (False, None)
        
        if self.verify_password(password, user.password_hash):
            session_token = self.create_session(user)
            return (True, session_token)
        
        return (False, None)
    
    def remove_user(self, user_id):
        """
        Removes user by ID and invalidates all sessions.
        
        Args:
            user_id (str): Unique identifier of the user
            
        Returns:
            bool: True if user was removed, False if not found
        """
        self.users = [u for u in self.users if u.id != user_id]
        self.cache.delete(f"user_sessions:{user_id}")
        self.db.delete_user(user_id)
        return True
    
    def get_user_by_email(self, email):
        """Retrieve user by email address with cache-first lookup."""
        cached = self.cache.get(f"user:{email}")
        if cached:
            return cached
        
        user = self.db.query_user_by_email(email)
        if user:
            self.cache.set(f"user:{email}", user, ex=self.session_timeout)
        return user
    
    def hash_password(self, password):
        """Hash password using bcrypt with salt."""
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    
    def verify_password(self, password, hash):
        """Verify password against stored hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode(), hash)
    
    def create_session(self, user):
        """Create new session token for authenticated user."""
        import uuid
        token = str(uuid.uuid4())
        self.cache.set(f"session:{token}", user.id, ex=self.session_timeout)
        return token
    
    def email_exists(self, email):
        """Check if email is already registered."""
        return any(u.email == email for u in self.users)


class DataProcessor:
    """
    Processes data pipelines including ETL operations.
    Handles data transformation, validation, and enrichment.
    """
    
    def __init__(self, config):
        """
        Initialize DataProcessor with configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - batch_size: Number of records per batch
                - max_retries: Maximum retry attempts
                - timeout: Processing timeout in seconds
        """
        self.config = config
        self.batch_size = config.get('batch_size', 1000)
        self.max_retries = config.get('max_retries', 3)
        self.processed_count = 0
    
    def process(self, data):
        """
        Main processing method that orchestrates the ETL pipeline.
        
        Args:
            data (list): Raw data records to process
            
        Returns:
            list: Transformed and validated data records
            
        Raises:
            ProcessingError: If processing fails after max retries
        """
        cleaned = self.clean_data(data)
        transformed = self.transform(cleaned)
        validated = self.validate(transformed)
        enriched = self.enrich(validated)
        
        self.processed_count += len(enriched)
        return enriched
    
    def clean_data(self, data):
        """Remove null values, duplicates, and invalid records."""
        cleaned = []
        seen_ids = set()
        
        for record in data:
            if not record or record.get('id') in seen_ids:
                continue
            seen_ids.add(record['id'])
            cleaned.append(record)
        
        return cleaned
    
    def transform(self, data):
        """Apply business logic transformations to data."""
        transformed = []
        for record in data:
            record['processed_at'] = self.get_timestamp()
            record['normalized_name'] = record.get('name', '').strip().lower()
            transformed.append(record)
        return transformed
    
    def validate(self, data):
        """Validate data against schema and business rules."""
        valid = []
        for record in data:
            if self.is_valid_record(record):
                valid.append(record)
        return valid
    
    def enrich(self, data):
        """Enrich data with additional information from external sources."""
        for record in data:
            record['enriched'] = True
            record['metadata'] = self.fetch_metadata(record['id'])
        return data
    
    def is_valid_record(self, record):
        """Check if record meets validation criteria."""
        required_fields = ['id', 'name', 'email']
        return all(field in record for field in required_fields)
    
    def fetch_metadata(self, record_id):
        """Fetch additional metadata for a record."""
        return {'source': 'api', 'version': '1.0'}
    
    def get_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """
    Main entry point for the application.
    Initializes components and starts the processing pipeline.
    """
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting application...")
    
    # Initialize database and cache connections
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'myapp',
        'user': 'admin',
        'password': 'secret'
    }
    
    cache_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    }
    
    # Create managers
    manager = UserManager(db_config, cache_config)
    processor_config = {
        'batch_size': 500,
        'max_retries': 5,
        'timeout': 30
    }
    processor = DataProcessor(processor_config)
    
    # Sample workflow
    try:
        # Add users
        user1 = manager.add_user("John Doe", "john@example.com", "securepass123")
        user2 = manager.add_user("Jane Smith", "jane@example.com", "strongpass456")
        
        logger.info(f"Created users: {user1.email}, {user2.email}")
        
        # Authenticate
        success, token = manager.authenticate("john@example.com", "securepass123")
        if success:
            logger.info(f"Authentication successful. Token: {token}")
        
        # Process data
        sample_data = [
            {'id': 1, 'name': 'Product A', 'email': 'product_a@company.com'},
            {'id': 2, 'name': 'Product B', 'email': 'product_b@company.com'},
            {'id': 3, 'name': 'Product C', 'email': 'product_c@company.com'}
        ]
        
        result = processor.process(sample_data)
        logger.info(f"Processed {len(result)} records")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise
    
    logger.info("Application completed successfully")


def helper_function(x, y):
    """
    Utility function for mathematical operations.
    
    Args:
        x (float): First operand
        y (float): Second operand
        
    Returns:
        float: Sum of x and y
    """
    return x + y


def calculate_metrics(data_points):
    """
    Calculate statistical metrics from data points.
    
    Args:
        data_points (list): List of numerical values
        
    Returns:
        dict: Dictionary containing mean, median, std_dev, min, max
    """
    if not data_points:
        return None
    
    import statistics
    
    return {
        'mean': statistics.mean(data_points),
        'median': statistics.median(data_points),
        'std_dev': statistics.stdev(data_points) if len(data_points) > 1 else 0,
        'min': min(data_points),
        'max': max(data_points),
        'count': len(data_points)
    }


if __name__ == "__main__":
    main()
