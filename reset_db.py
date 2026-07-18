from database import engine
import models

models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)
print('✅ Database reset & updated with new columns!')