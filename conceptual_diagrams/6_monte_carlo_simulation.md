```mermaid
graph TD
    A[Business Impact Model] --> B[Parameter Distribution Definition]
    B --> C[Simulation Engine]
    
    C --> D[Revenue Simulation]
    C --> E[Cost Simulation]
    C --> F[Market Share Simulation]
    C --> G[Operational Simulation]
    
    D --> H[Distribution Analysis]
    E --> H
    F --> H
    G --> H
    
    H --> I[VaR Calculation]
    H --> J[Confidence Intervals]
    H --> K[Stress Testing]
    
    style H fill:#f96,stroke:#333
```

Key Points:
1. Simulates variations in business metrics
2. Uses parameter distributions from historical data
3. Runs multiple scenarios simultaneously
4. Calculates risk metrics (VaR, etc.)
5. Identifies extreme scenarios