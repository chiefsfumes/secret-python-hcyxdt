```mermaid
graph TD
    A[Abstract Impact Scores] --> B[Impact Matrices]
    B --> C{Risk Category}
    C -->|Physical Risk| D[Revenue Matrix]
    C -->|Transition Risk| E[Margin Matrix]
    C -->|Nature Risk| F[Market Share Matrix]
    
    D --> G[Business Metrics]
    E --> G
    F --> G
    
    G --> H[Revenue Impact]
    G --> I[Cost Impact]
    G --> J[Market Share Impact]
    G --> K[Operational Impact]
    G --> L[Supply Chain Impact]

    style B fill:#f96,stroke:#333
```

Key Points:
1. Translation happens after time series but before scenarios
2. Category-specific impact matrices
3. Multiple business metrics considered
4. Maintains traceability to risk categories
5. Allows for category-specific adjustments