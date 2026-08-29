import React from 'react';
import { View, Text, StyleSheet, ScrollView, SafeAreaView, Dimensions } from 'react-native';
import { BarChart, LineChart } from 'react-native-chart-kit';
import { useRoute } from '@react-navigation/native';

const screenWidth = Dimensions.get('window').width;

export default function DashboardScreen() {
  const route = useRoute<any>();
  // Extract JSON response passed via React Navigation params
  const dashboardData = route.params?.dashboardData || {};

  // 1. Calories Burned extraction with fallback
  const caloriesBurned = dashboardData.calories_burned ?? 0;

  // 2. SHAP Graph extraction with fallback
  const shapData = dashboardData.shap_graph_data ?? {
    labels: ['MET', 'Temp', 'Weight', 'Humid'],
    datasets: [
      {
        data: [35, 30, 20, 15] // Static Placeholder: Will be replaced by live dynamic variables once ML backend is built
      }
    ]
  };

  // Ensure labels are shortened for mobile screens (applies to live dynamic data too)
  if (shapData.labels) {
    shapData.labels = shapData.labels.map((label: string) => {
      if (label === 'MET (Pacing)') return 'MET';
      if (label === 'Temperature') return 'Temp';
      if (label === 'Humidity') return 'Humid';
      return label;
    });
  }

  if (shapData.datasets && shapData.datasets[0] && shapData.datasets[0].data) {
    shapData.datasets[0].data = shapData.datasets[0].data.map((val: number) => {
      return (val > 0 && val <= 1) ? Math.round(val * 100) : Math.round(val);
    });
  }

  // 3. Circadian Forecast curve extraction with fallback
  const fatigueData = {
    labels: ['Now', '+1h', '+2h', '+3h', '+4h'],
    datasets: [
      {
        data: dashboardData.fatigue_curve ?? [20, 35, 60, 85, 95], // Fallback Fatigue level
      }
    ],
    legend: ['Predicted Fatigue Crash']
  };

  // Re-inject the functional styling properties since functions cannot be transmitted via JSON over network
  if (fatigueData.datasets && fatigueData.datasets.length > 0) {
    fatigueData.datasets[0].color = (opacity = 1) => `rgba(255, 59, 48, ${opacity})`;
    fatigueData.datasets[0].strokeWidth = 3;
  }

  const chartConfig = {
    backgroundGradientFrom: '#ffffff',
    backgroundGradientTo: '#ffffff',
    color: (opacity = 1) => `rgba(0, 123, 255, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(51, 51, 51, ${opacity})`,
    strokeWidth: 2,
    barPercentage: 0.5,
    useShadowColorFromDataset: false,
    decimalPlaces: 0,
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        
        <View style={styles.headerContainer}>
          <Text style={styles.title}>Predictive Dashboard</Text>
          <Text style={styles.subtitle}>Stage 4: Analytics</Text>
        </View>

        <View style={styles.caloriesContainer}>
          <Text style={styles.caloriesLabel}>CALORIES BURNED (kcal)</Text>
          <Text style={styles.caloriesValue}>{caloriesBurned}</Text>
        </View>

        <View style={styles.chartContainer}>
          <Text style={styles.chartTitle}>SHAP Explainability Graph</Text>
          <Text style={styles.chartSubtitle}>Feature Importance</Text>
          <BarChart
            data={shapData}
            width={screenWidth - 48}
            height={220}
            yAxisLabel=""
            yAxisSuffix="%"
            fromZero={true}
            segments={4}
            formatYLabel={(y) => Math.round(Number(y)).toString()}
            chartConfig={chartConfig}
            verticalLabelRotation={30}
            style={styles.chartStyle}
            showValuesOnTopOfBars
          />
        </View>

        <View style={styles.chartContainer}>
          <Text style={styles.chartTitle}>Circadian Fatigue Forecaster</Text>
          <Text style={styles.chartSubtitle}>Crash Curve Over Time</Text>
          <LineChart
            data={fatigueData}
            width={screenWidth - 48}
            height={220}
            chartConfig={{
              ...chartConfig,
              color: (opacity = 1) => `rgba(255, 59, 48, ${opacity})`, // Red accent for fatigue
            }}
            bezier
            style={styles.chartStyle}
          />
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F9FF',
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  headerContainer: {
    marginTop: 20,
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 4,
  },
  caloriesContainer: {
    backgroundColor: '#007BFF',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 24,
    shadowColor: '#007BFF',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  caloriesLabel: {
    color: '#E6F0FA',
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 1,
    marginBottom: 8,
  },
  caloriesValue: {
    color: '#ffffff',
    fontSize: 48,
    fontWeight: 'bold',
  },
  chartContainer: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 5,
    elevation: 3,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  chartSubtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 16,
  },
  chartStyle: {
    marginVertical: 8,
    borderRadius: 16,
    marginLeft: -16, // Adjusting padding to center chart visually
  },
});
