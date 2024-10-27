```mermaid
graph TD
    A[Base Business Impact] --> B{Scenario Type}
    
    B -->|Net Zero 2050| C[Lower Physical Impact]
    B -->|Net Zero 2050| D[Higher Transition Impact]
    
    B -->|Delayed Transition| E[Higher Physical Impact]
    B -->|Delayed Transition| F[Much Higher Transition Impact]
    
    C --> G[Adjusted Business Metrics]
    D --> G
    E --> G
    F --> G
    
    style G fill:#f96,stroke:#333
```

Key Points:
1. Scenario-specific adjustments to business impacts
2. Different effects by risk category
3. Maintains business metric relationships
4. Allows for non-linear adjustments
5. Scenario-specific propagation changes