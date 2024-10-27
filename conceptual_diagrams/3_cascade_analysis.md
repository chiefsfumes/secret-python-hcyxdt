```mermaid
graph TD
    A[Initial Risk] --> B[Calculate Direct Business Impact]
    B --> C[Build Cascade Tree]
    C --> D[Propagation Engine]
    D --> E[Calculate Delays]
    D --> F[Calculate Strengths]
    
    E --> G[Time-Adjusted Impact]
    F --> G
    G --> H[Next Level Risks]
    H -->|Recurse| C
    
    style G fill:#f96,stroke:#333
```

Key Points:
1. Starts with direct business impacts
2. Builds tree of cascading effects
3. Considers propagation delays
4. Calculates interaction strengths
5. Recursive propagation with depth limit
6. Time-adjusted impacts at each level