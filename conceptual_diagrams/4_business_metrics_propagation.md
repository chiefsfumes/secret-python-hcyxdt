```mermaid
graph TD
    A[Source Risk Impact] --> B[Propagation Strength]
    A --> C[Time Delay]
    
    B --> D[Revenue Decay]
    B --> E[Cost Amplification]
    B --> F[Market Share Erosion]
    B --> G[Operational Degradation]
    
    C --> H[Time-Based Decay]
    
    D --> I[Propagated Business Impact]
    E --> I
    F --> I
    G --> I
    H --> I

    style I fill:#f96,stroke:#333
```

Key Points:
1. Each business metric propagates differently
2. Time delays affect propagation strength
3. Different decay rates per metric
4. Combined propagated impact
5. Maintains metric relationships