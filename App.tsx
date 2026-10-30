import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons'; // Make sure expo/vector-icons is available

// Import Screens
import DataInputScreen from './screens/DataInputScreen';
import DashboardScreen from './screens/DashboardScreen';
import HistoryScreen from './screens/HistoryScreen';
import ProfileScreen from './screens/ProfileScreen';

export type RootStackParamList = {
  Tabs: undefined;
};

export type TabParamList = {
  Profile: undefined;
  Tracker: undefined;
  Dashboard: { dashboardData?: any };
  History: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<TabParamList>();

function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerStyle: { backgroundColor: '#F5F9FF' },
        headerShadowVisible: false,
        headerTintColor: '#333',
        headerTitleStyle: { fontWeight: 'bold' },
        tabBarIcon: ({ color, size }) => {
          let iconName: any;
          if (route.name === 'Profile') iconName = 'person';
          else if (route.name === 'Tracker') iconName = 'mic';
          else if (route.name === 'Dashboard') iconName = 'stats-chart';
          else if (route.name === 'History') iconName = 'time';
          return <Ionicons name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#007BFF',
        tabBarInactiveTintColor: 'gray',
      })}
    >
      <Tab.Screen name="Profile" component={ProfileScreen} />
      <Tab.Screen name="Tracker" component={DataInputScreen} options={{ title: 'NEAT Tracker' }} />
      <Tab.Screen name="Dashboard" component={DashboardScreen} options={{ title: 'Analytics' }} />
      <Tab.Screen name="History" component={HistoryScreen} />
    </Tab.Navigator>
  );
}

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Tabs" component={TabNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
